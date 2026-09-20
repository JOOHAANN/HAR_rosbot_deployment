
# Naming Specification

## Configuration Files

```
<core algorithm>-<backbone>-<dataset>-<modality>-<data format>-<sample strategy>-<clip_len>x<frame_interval>x<num_clips>
```

For example, the configuration file `tsn_r50_ucf101_rgb_raw_dense_1x16x4`

* `core algorithm: tsn`
* `backbone: r50(resnet-50)`
* `dataset: ucf101`
* `modality: rgb`
* `data format: raw(raw frame)`
* `sample strategy: dense(dense sample)`
* `clip_len: 1`
* `frame_interval: 16`
* `num_clips: 4`