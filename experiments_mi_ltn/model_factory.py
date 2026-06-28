"""Model factory for incremental MI experiment integrations."""

from __future__ import annotations

from typing import Any

from mi_ltn_common import CHUNK_SIZE, NUM_CLASSES, NUM_ELECTRODES, load_eegnet_class


def build_model(
    model_name,
    n_chans,
    n_outputs,
    n_times,
    sfreq=None,
    chs_info=None,
    model_kwargs=None,
):
    """Build a supported EEG model without changing the existing EEGNet path."""

    normalized_name = str(model_name).strip().lower()
    model_kwargs = dict(model_kwargs or {})

    n_chans = NUM_ELECTRODES if n_chans is None else n_chans
    n_outputs = NUM_CLASSES if n_outputs is None else n_outputs
    n_times = CHUNK_SIZE if n_times is None else n_times

    if normalized_name == "eegnet":
        return _build_eegnet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, model_kwargs=model_kwargs)
    if normalized_name == "eegnex":
        return _build_eegnex(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=chs_info,
            model_kwargs=model_kwargs,
        )

    raise ValueError(
        f"Unsupported model_name={model_name!r}. Supported values: 'eegnet', 'eegnex'."
    )


def _build_eegnet(
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
    model_kwargs: dict[str, Any],
):
    EEGNet = load_eegnet_class()
    eegnet_kwargs = {
        "chunk_size": n_times,
        "num_electrodes": n_chans,
        "num_classes": n_outputs,
    }
    eegnet_kwargs.update(model_kwargs)
    return EEGNet(**eegnet_kwargs)


def _build_eegnex(
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
    sfreq,
    chs_info,
    model_kwargs: dict[str, Any],
):
    from braindecode.models import EEGNeX

    eegnex_kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
    }
    if sfreq is not None:
        eegnex_kwargs["sfreq"] = sfreq
    if chs_info is not None:
        eegnex_kwargs["chs_info"] = chs_info
    eegnex_kwargs.update(model_kwargs)
    return EEGNeX(**eegnex_kwargs)
