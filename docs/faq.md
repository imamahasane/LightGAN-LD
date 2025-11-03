# FAQ
- Unstable early training? Use the default OneCycle warm-up; ensure batch >= 32 if possible.
- SWA not helping? Increase anneal epochs; start >= epoch 100.
- OOM? Reduce base_channels or disable CondConv/ECA; lower batch size.
