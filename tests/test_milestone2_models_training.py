"""Unit test suite for Milestone 2 model architectures, losses, calibration, and trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from mri_tumor.models import (
    SimpleCNN,
    TimmClassifier,
    build_model,
    build_model_from_config,
    resolve_target_conv_layer,
)
from mri_tumor.training import (
    ABSTAIN_LABEL,
    FocalLoss,
    TemperatureScaler,
    Trainer,
    apply_temperature_and_abstention,
    build_loss_fn,
    compute_class_weights,
    find_sensitivity_threshold,
    load_checkpoint,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_simple_cnn_and_timm_forward_and_target_layer():
    """Verifies logits, feature map shapes, and Grad-CAM target layers for CNN models."""
    x = torch.randn(2, 3, 224, 224)

    simple = SimpleCNN(num_classes=2, dropout=0.2)
    simple_feats = simple.forward_features(x)
    simple_logits = simple(x)
    assert simple_feats.shape == (2, 256, 14, 14)
    assert simple_logits.shape == (2, 2)
    simple_target = simple.get_target_layer()
    assert isinstance(simple_target, nn.Conv2d)
    assert simple_target.out_channels == 256

    eff = TimmClassifier("efficientnet_b0", num_classes=2, pretrained=False)
    eff_feats = eff.forward_features(x)
    eff_logits = eff(x)
    assert eff_feats.shape[:2] == (2, eff.feature_dim)
    assert eff_feats.ndim == 4
    assert eff_logits.shape == (2, 2)
    assert eff.get_target_layer() is eff.backbone.conv_head

    res = TimmClassifier("resnet50", num_classes=3, pretrained=False)
    res_feats = res.forward_features(x)
    res_logits = res(x)
    assert res_feats.shape[:2] == (2, res.feature_dim)
    assert res_logits.shape == (2, 3)
    assert res.get_target_layer() is res.backbone.layer4[-1]

    with pytest.raises(ValueError, match="no nn.Conv2d found"):
        resolve_target_conv_layer(nn.Sequential(nn.Linear(8, 4)))


def test_two_phase_freeze_unfreeze_and_bn_eval():
    """Verifies backbone freezing disables grads, locks backbone BN in eval, and keeps head trainable."""
    model = SimpleCNN(num_classes=3, dropout=0.2)

    model.freeze_backbone()
    assert model._backbone_frozen is True
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    # Switching outer model to train() must keep backbone and its BatchNorms in eval mode
    model.train()
    assert model.training is True
    assert model.head.training is True
    assert model.backbone.training is False
    bn_layers = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) == 4
    assert all(not bn.training for bn in bn_layers)

    # Verify BN running stats do not drift on forward pass while frozen
    running_mean_before = bn_layers[0].running_mean.clone()
    _ = model(torch.randn(4, 3, 64, 64) * 50.0 + 100.0)
    assert torch.equal(bn_layers[0].running_mean, running_mean_before)

    # Unfreeze restores gradients and train mode
    model.unfreeze_backbone()
    assert model._backbone_frozen is False
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert model.backbone.training is True
    assert all(bn.training for bn in bn_layers)

    # Also verify TimmClassifier freeze/unfreeze behavior
    timm_model = TimmClassifier("efficientnet_b0", num_classes=3, pretrained=False)
    timm_model.freeze_backbone()
    timm_model.train()
    assert timm_model._backbone_frozen is True
    assert timm_model.backbone.training is False
    assert timm_model.head.training is True
    assert all(not p.requires_grad for p in timm_model.backbone.parameters())
    assert all(p.requires_grad for p in timm_model.head.parameters())
    timm_model.unfreeze_backbone()
    assert timm_model.backbone.training is True
    assert all(p.requires_grad for p in timm_model.backbone.parameters())


def test_model_factory_and_yaml_configs():
    """Loads stage1.yaml and stage2.yaml, overrides pretrained=False, and builds models."""
    with open(REPO_ROOT / "configs" / "stage1.yaml", "r", encoding="utf-8") as f:
        cfg1 = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "stage2.yaml", "r", encoding="utf-8") as f:
        cfg2 = yaml.safe_load(f)

    cfg1["model"]["pretrained"] = False
    cfg2["model"]["pretrained"] = False

    m1 = build_model_from_config(cfg1)
    assert isinstance(m1, TimmClassifier)
    assert m1.num_classes == 2
    assert m1.backbone_name == cfg1["model"]["backbone"]

    m2 = build_model_from_config(cfg2)
    assert isinstance(m2, TimmClassifier)
    assert m2.num_classes == 3
    assert m2.backbone_name == cfg2["model"]["backbone"]

    m_res = build_model("cnn", backbone="resnet50", num_classes=3, pretrained=False)
    assert isinstance(m_res, TimmClassifier)
    assert m_res.backbone_name == "resnet50"
    assert m_res.num_classes == 3

    m_simple = build_model("cnn", backbone="simple_cnn", num_classes=2)
    assert isinstance(m_simple, SimpleCNN)

    m_simple_direct = build_model("simple_cnn", num_classes=4)
    assert isinstance(m_simple_direct, SimpleCNN)
    assert m_simple_direct.num_classes == 4

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        build_model("unknown_arch")


def test_losses_weighted_ce_and_focal():
    """Verifies balanced class weights, FocalLoss gamma damping, and build_loss_fn."""
    labels = [0] * 90 + [1] * 10
    weights = compute_class_weights(labels)
    assert weights.shape == (2,)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(weights[1] / weights[0], torch.tensor(9.0), atol=1e-4)

    logits = torch.tensor([[3.0, 0.1, -1.0], [0.2, 2.5, 0.1]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 1], dtype=torch.long)

    ce = nn.CrossEntropyLoss()
    fl0 = FocalLoss(gamma=0.0, alpha=None, label_smoothing=0.0)
    fl2 = FocalLoss(gamma=2.0, alpha=None, label_smoothing=0.0)

    loss_ce = ce(logits, targets)
    loss_fl0 = fl0(logits, targets)
    loss_fl2 = fl2(logits, targets)

    assert torch.isclose(loss_fl0, loss_ce, atol=1e-6)
    assert loss_fl2 < loss_fl0  # Gamma=2 damps well-classified samples

    loss_fn_wce = build_loss_fn("weighted_ce", class_weights=weights, label_smoothing=0.05)
    loss_fn_focal = build_loss_fn("focal", class_weights=weights, label_smoothing=0.05, focal_gamma=2.0)

    out_wce = loss_fn_wce(logits[:, :2], torch.tensor([0, 1]))
    out_focal = loss_fn_focal(logits[:, :2], torch.tensor([0, 1]))
    assert torch.isfinite(out_wce) and out_wce.item() > 0.0
    assert torch.isfinite(out_focal) and out_focal.item() > 0.0
    out_focal.backward()
    assert logits.grad is not None

    with pytest.raises(ValueError, match="Unsupported loss function"):
        build_loss_fn("dice_loss")


def test_stage1_sensitivity_threshold_calibration():
    """Verifies sensitivity-constrained threshold search and fallback when target is unreachable."""
    # 50 positives (probs 0.30..0.99), 50 negatives (probs 0.01..0.50)
    # Only 1 positive is at 0.30, remaining 49 positives >= 0.45 => 49/50 = 0.98 sensitivity at 0.45!
    pos_probs = np.concatenate([np.array([0.30]), np.linspace(0.45, 0.99, 49)])
    neg_probs = np.concatenate([np.linspace(0.01, 0.40, 45), np.linspace(0.46, 0.70, 5)])
    probs = np.concatenate([pos_probs, neg_probs])
    targets = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

    res = find_sensitivity_threshold(probs, targets, target_sensitivity=0.98)
    assert res.achieved_target is True
    assert res.sensitivity >= 0.98
    assert np.isclose(res.threshold, 0.45, atol=1e-6)
    assert np.isclose(res.specificity, 45 / 50, atol=1e-6)

    # Unreachable target when restricted to candidate thresholds >= 0.50
    res_fallback = find_sensitivity_threshold(
        probs,
        targets,
        target_sensitivity=0.98,
        min_threshold=0.50,
    )
    assert res_fallback.achieved_target is False
    assert res_fallback.sensitivity < 0.98
    assert res_fallback.threshold >= 0.50


def test_stage2_temperature_scaling_and_abstention():
    """Verifies TemperatureScaler reduces NLL on overconfident logits and abstention flags low confidence."""
    rng = np.random.RandomState(42)
    n_samples = 300
    true_logits = rng.randn(n_samples, 3).astype(np.float32) * 1.5
    true_probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=p) for p in true_probs], dtype=np.int64)

    # Multiply logits by 4.0 to simulate severe overconfidence
    overconfident_logits = true_logits * 4.0
    nll_before = nn.CrossEntropyLoss()(
        torch.tensor(overconfident_logits),
        torch.tensor(labels),
    ).item()

    scaler = TemperatureScaler(init_temperature=1.0)
    learned_temp = scaler.fit(overconfident_logits, labels, max_iter=50, lr=0.01)

    assert learned_temp > 1.5
    with torch.no_grad():
        scaled_logits = scaler(torch.tensor(overconfident_logits))
        nll_after = nn.CrossEntropyLoss()(scaled_logits, torch.tensor(labels)).item()
    assert nll_after < nll_before

    # Test abstention logic at uncertainty_threshold=0.70
    sample_logits = np.array([
        [6.0, 0.0, 0.0],  # High confidence -> glioma
        [0.2, 0.1, 0.0],  # Low confidence -> abstain
        [0.0, 5.0, 0.1],  # High confidence -> meningioma
    ], dtype=np.float32)
    out = apply_temperature_and_abstention(
        logits=sample_logits,
        temperature=1.0,
        uncertainty_threshold=0.70,
        class_names=("glioma", "meningioma", "pituitary"),
    )
    assert out["probs"].shape == (3, 3)
    assert out["pred_indices"].tolist() == [0, 0, 1]
    assert out["abstain_mask"].tolist() == [False, True, False]
    assert out["labels"] == ["glioma", ABSTAIN_LABEL, "meningioma"]


def test_trainer_two_phase_training_and_checkpoint_reload(tmp_path: Path):
    """Runs 4-epoch training with 2 frozen epochs, verifies freeze schedule, checkpoint reload, and predict_logits."""
    set_seed(42)
    num_samples = 24
    images = torch.randn(num_samples, 3, 64, 64)
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(images[:16], labels[:16]), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(images[16:], labels[16:]), batch_size=8, shuffle=False)

    model = SimpleCNN(num_classes=2, dropout=0.1)
    initial_backbone_weight = model.backbone[0][0].weight.detach().clone()

    # Track backbone freeze state and weight snapshot at the end of each epoch via hook
    epoch_freeze_states: list[bool] = []
    epoch_backbone_changed: list[bool] = []

    trainer = Trainer(
        model=model,
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=4,
        lr=1e-2,
        warmup_epochs=1,
        freeze_backbone_epochs=2,
        early_stopping_patience=10,
        checkpoint_dir=tmp_path / "ckpt",
        use_amp=True,  # Must stay clean FP32 on CPU
    )

    orig_eval = trainer.evaluate

    def _spy_evaluate(loader: DataLoader) -> tuple[float, float]:
        epoch_freeze_states.append(bool(model._backbone_frozen))
        changed = not torch.allclose(model.backbone[0][0].weight.detach(), initial_backbone_weight)
        epoch_backbone_changed.append(changed)
        return orig_eval(loader)

    trainer.evaluate = _spy_evaluate  # type: ignore[method-assign]

    history = trainer.fit(train_loader, val_loader)

    assert len(history.train_loss) == 4
    assert len(history.val_loss) == 4
    assert len(history.lr) == 4
    assert epoch_freeze_states == [True, True, False, False]
    assert epoch_backbone_changed[:2] == [False, False]
    assert epoch_backbone_changed[2:] == [True, True]

    assert history.best_checkpoint_path is not None
    ckpt_path = Path(history.best_checkpoint_path)
    assert ckpt_path.exists()

    # Reload best checkpoint into a fresh model and verify exact parameter and prediction parity
    reloaded_model = SimpleCNN(num_classes=2, dropout=0.1)
    meta = load_checkpoint(ckpt_path, reloaded_model, device="cpu")
    assert meta["epoch"] == history.best_epoch

    trainer.model.load_state_dict(reloaded_model.state_dict())
    logits, targets = trainer.predict_logits(val_loader)
    assert logits.shape == (8, 2)
    assert targets.shape == (8,)

    reloaded_model.eval()
    with torch.no_grad():
        direct_logits = reloaded_model(images[16:]).numpy()
    assert np.allclose(logits, direct_logits, atol=1e-6)

    # Also verify Trainer scheduler does not crash when epochs <= warmup_epochs
    short_trainer = Trainer(
        model=SimpleCNN(num_classes=2),
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=1,
        warmup_epochs=2,
        freeze_backbone_epochs=0,
        checkpoint_dir=tmp_path / "short_ckpt",
    )
    short_history = short_trainer.fit(train_loader, val_loader)
    assert len(short_history.train_loss) == 1


def test_gradcam_backward_hooks_across_backbones():
    """Verifies get_target_layer() produces activations and grads without inplace hook errors."""
    x = torch.randn(1, 3, 128, 128)
    models = [
        SimpleCNN(num_classes=2),
        TimmClassifier("efficientnet_b0", num_classes=2, pretrained=False),
        TimmClassifier("resnet18", num_classes=2, pretrained=False),
        TimmClassifier("resnet50", num_classes=3, pretrained=False),
        TimmClassifier("mobilenetv3_large_100", num_classes=2, pretrained=False),
        TimmClassifier("densenet121", num_classes=2, pretrained=False),
        TimmClassifier("vgg16", num_classes=2, pretrained=False),
    ]

    for model in models:
        model.eval()
        target = model.get_target_layer()
        acts: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        h_fwd = target.register_forward_hook(lambda _m, _i, out: acts.append(out))
        h_bwd = target.register_full_backward_hook(lambda _m, _gi, gout: grads.append(gout[0]))
        try:
            logits = model(x)
            logits[0, 0].backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

        assert len(acts) == 1 and acts[0].ndim == 4
        assert len(grads) == 1 and grads[0].shape == acts[0].shape
        assert torch.isfinite(grads[0]).all()


def test_model_and_factory_edge_case_guards():
    """Verifies shape and parameter validation guards across models and factory."""
    with pytest.raises(ValueError, match="num_classes"):
        SimpleCNN(num_classes=0)
    with pytest.raises(ValueError, match="dropout"):
        TimmClassifier("efficientnet_b0", pretrained=False, dropout=1.0)
    with pytest.raises(ValueError, match="model config dict"):
        build_model_from_config({"model": None})
    with pytest.raises(ValueError, match="Expected config dict"):
        build_model_from_config("invalid")  # type: ignore[arg-type]


def test_losses_edge_cases_and_amp_dtypes():
    """Verifies FocalLoss shape guards, alpha validation, list weights, and fp16/bf16 stability."""
    fl = FocalLoss(gamma=2.0, alpha=[0.5, 1.5, 1.0])
    logits_fp16 = torch.tensor([[2.0, -1.0, 0.5], [0.1, 1.5, -0.5]], dtype=torch.float16)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss_fp16 = fl(logits_fp16, targets)
    assert torch.isfinite(loss_fp16)

    loss_fn = build_loss_fn("focal", class_weights=[1.0, 2.0, 0.5], focal_gamma=1.5)
    assert torch.isfinite(loss_fn(logits_fp16.float(), targets))

    with pytest.raises(ValueError, match="2D logits"):
        fl(torch.randn(3), torch.tensor(0))
    with pytest.raises(ValueError, match="1D targets"):
        fl(torch.randn(2, 3), torch.tensor([[0], [1]]))
    with pytest.raises(ValueError, match="alpha length"):
        FocalLoss(alpha=[1.0, 2.0])(torch.randn(2, 3), targets)
    with pytest.raises(ValueError, match="scalar or 1D"):
        FocalLoss(alpha=[[1.0, 2.0]])


def test_calibration_edge_cases_and_temperature_clamping():
    """Verifies min_threshold filtering with explicit grid, validation errors, and T clamping."""
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    targets = np.array([0, 1, 1, 1])

    res = find_sensitivity_threshold(
        probs,
        targets,
        target_sensitivity=0.98,
        thresholds=[0.1, 0.3, 0.5, 0.8],
        min_threshold=0.45,
    )
    assert res.threshold >= 0.45

    with pytest.raises(ValueError, match="min_threshold"):
        find_sensitivity_threshold(probs, targets, thresholds=[0.1, 0.2], min_threshold=0.5)
    with pytest.raises(ValueError, match="binary"):
        find_sensitivity_threshold(probs, np.array([0, 1, 2, 1]))
    with pytest.raises(ValueError, match="finite"):
        find_sensitivity_threshold(np.array([0.2, np.nan, 0.8, 0.9]), targets)

    # Anti-calibrated logits push T upward; scaler must stay clamped within max_temperature
    bad_logits = torch.tensor(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]] * 10
    )
    bad_labels = torch.tensor([1, 2, 0] * 10)
    scaler_hi = TemperatureScaler(min_temperature=0.05, max_temperature=25.0)
    t_hi = scaler_hi.fit(bad_logits, bad_labels, max_iter=50, lr=0.5)
    assert 0.05 <= t_hi <= 25.0

    # Separable logits push T downward; scaler must stay clamped above min_temperature
    sep_logits = torch.tensor(
        [[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]] * 10
    )
    sep_labels = torch.tensor([0, 1, 2] * 10)
    scaler_lo = TemperatureScaler(min_temperature=0.05, max_temperature=25.0)
    t_lo = scaler_lo.fit(sep_logits, sep_labels, max_iter=50, lr=0.5)
    assert 0.05 <= t_lo <= 25.0

    # Half precision forward preserves dtype and stays finite
    out_fp16 = scaler_hi(sep_logits.half())
    assert out_fp16.dtype == torch.float16
    assert torch.isfinite(out_fp16).all()

    with pytest.raises(ValueError, match="positive and finite"):
        TemperatureScaler(init_temperature=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        scaler_hi.fit(np.array([[np.nan, 0.0]]), np.array([0]))
    with pytest.raises(ValueError, match="positive and finite"):
        apply_temperature_and_abstention(np.zeros((1, 3)), temperature=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        apply_temperature_and_abstention(np.array([[np.nan, 0.0, 0.0]]))


def test_trainer_restores_best_weights_and_validates_args(tmp_path: Path):
    """Verifies Trainer restores best_model.pt weights after worsening epochs and validates args."""
    set_seed(7)
    train_ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.tensor([0, 1] * 4))
    val_ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.tensor([0, 1] * 4))
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = SimpleCNN(num_classes=2, dropout=0.0)
    trainer = Trainer(
        model=model,
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=5,
        lr=1e-2,
        warmup_epochs=1,
        freeze_backbone_epochs=0,
        early_stopping_patience=2,
        checkpoint_dir=tmp_path / "restore_ckpt",
    )

    # Inject deterministic worsening val_loss after epoch 1 so early stopping triggers at epoch 3
    fake_val_losses = iter([0.50, 0.20, 0.40, 0.60, 0.80])
    trainer.evaluate = lambda _loader: (next(fake_val_losses), 0.5)  # type: ignore[method-assign]

    history = trainer.fit(train_loader, val_loader)
    assert history.best_epoch == 1
    assert len(history.val_loss) == 4  # Stopped early after 2 non-improving epochs (2, 3)
    assert history.best_checkpoint_path is not None

    # Trainer.model must already match best_model.pt (epoch 1), not epoch 3
    fresh = SimpleCNN(num_classes=2, dropout=0.0)
    load_checkpoint(history.best_checkpoint_path, fresh, device="cpu")
    for p_trained, p_best in zip(trainer.model.parameters(), fresh.parameters()):
        assert torch.allclose(p_trained, p_best)

    # Verify LR rises during warmup and decays afterward
    set_seed(7)
    lr_model = SimpleCNN(num_classes=2)
    lr_trainer = Trainer(
        model=lr_model,
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=4,
        lr=1e-2,
        warmup_epochs=2,
        freeze_backbone_epochs=0,
        early_stopping_patience=0,
        checkpoint_dir=tmp_path / "lr_ckpt",
    )
    lr_hist = lr_trainer.fit(train_loader, val_loader)
    assert lr_hist.lr[0] < lr_hist.lr[1]
    assert lr_hist.lr[3] < lr_hist.lr[2]

    with pytest.raises(ValueError, match="epochs"):
        Trainer(model=SimpleCNN(), criterion=build_loss_fn("ce"), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        Trainer(model=SimpleCNN(), criterion=build_loss_fn("ce"), lr=-1e-3)


def test_trainer_verbose_logging_and_quiet_mode(tmp_path: Path, capsys):
    """Verifies Trainer prints per-epoch status + early stopping when verbose=True and suppresses when False."""
    set_seed(3)
    train_ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.tensor([0, 1] * 4))
    val_ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.tensor([0, 1] * 4))
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    verbose_trainer = Trainer(
        model=SimpleCNN(num_classes=2, dropout=0.0),
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=4,
        lr=1e-2,
        warmup_epochs=1,
        freeze_backbone_epochs=1,
        early_stopping_patience=1,
        checkpoint_dir=tmp_path / "verbose_ckpt",
        verbose=True,
    )
    fake_losses = iter([0.30, 0.50, 0.60, 0.70])
    verbose_trainer.evaluate = lambda _loader: (next(fake_losses), 0.75)  # type: ignore[method-assign]
    capsys.readouterr()
    verbose_trainer.fit(train_loader, val_loader)
    out = capsys.readouterr().out
    assert "Epoch 01/04 [frozen]" in out
    assert "[best]" in out
    assert "Epoch 02/04 [unfrozen]" in out
    assert "Early stopping triggered at epoch 2" in out

    quiet_trainer = Trainer(
        model=SimpleCNN(num_classes=2, dropout=0.0),
        criterion=build_loss_fn("ce"),
        device="cpu",
        epochs=2,
        lr=1e-2,
        warmup_epochs=0,
        freeze_backbone_epochs=0,
        early_stopping_patience=0,
        checkpoint_dir=tmp_path / "quiet_ckpt",
        verbose=False,
    )
    capsys.readouterr()
    quiet_trainer.fit(train_loader, val_loader)
    quiet_out = capsys.readouterr().out
    assert quiet_out == ""
