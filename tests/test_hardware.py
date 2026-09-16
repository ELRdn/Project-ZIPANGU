from scripts.hardware_probe import classify_gpu, select_primary_gpu


def test_rx7600_wins_over_igpu_and_igpu_is_excluded():
    igpu = classify_gpu("AMD Radeon 780M Graphics", 512 * 1024 * 1024)
    rx7600 = classify_gpu("AMD Radeon RX 7600", 8 * 1024**3)
    primary = select_primary_gpu([igpu, rx7600])
    assert igpu["device_role"] == "iGPU"
    assert igpu["exclude_from_primary_benchmark"] is True
    assert primary == rx7600


def test_non_target_discrete_gpu_is_not_called_rx7600():
    gpu = classify_gpu("NVIDIA GeForce RTX 4090", 24 * 1024**3)
    assert gpu["device_role"] == "discrete_gpu"
    assert gpu["target_rx7600"] is False
    assert select_primary_gpu([gpu]) == gpu
