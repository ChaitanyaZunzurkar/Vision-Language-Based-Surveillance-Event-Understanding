# Model artifacts

Keep trained weights and other ML artifacts outside `server/` and `frontend/`.

```text
models/
  detection/
  anomaly/
  event_understanding/
```

The tracked YOLO checkpoint is under `models/detection/`. Large artifacts
should use Git LFS or an external model/artifact registry.
