# FireWatch-AI — Per-Model Comparison

**Test set:** 2 images (1 fire, 1 no_fire)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Status |
|-------|----------|-----------|--------|----|---------|--------|
| EfficientNet-B0 | 50.0% | 0.0% | 0.0% | 0.0% | 1.0 | EVALUATED |
| Heuristic | 100.0% | 100.0% | 100.0% | 100.0% | N/A | EVALUATED |
| YOLOv8 | — | — | — | — | — | NOT EVALUATED |
| CNN-LSTM | — | — | — | — | — | NOT EVALUATED |

## Model Audit Summary

### efficientnet
- **architecture**: EfficientNet-B0 (binary fire/no_fire head)
- **weights_path**: efficientnet_fire_best.pth
- **weight_status**: custom_fire_weights
- **can_detect_fire**: True
- **evaluation_status**: EVALUATED

### yolov8
- **architecture**: YOLOv8n
- **weights_path**: yolov8n.pt
- **weight_status**: coco_generic_weights
- **has_fire_classes**: False
- **can_detect_fire**: False
- **evaluation_status**: NOT_EVALUATED

### cnn_lstm
- **architecture**: EfficientNet-B0 CNN + LSTM (seq_length=5)
- **weights_path**: cnn_lstm_best.pth
- **weight_status**: random_weights
- **can_detect_fire**: False
- **evaluation_status**: NOT_EVALUATED
- **note**: CNN-LSTM requires temporal sequences of frames from the same scene. Still-image datasets cannot substitute for genuine temporal sequences.

### heuristic
- **architecture**: Color-histogram + brightness + texture heuristic
- **weights_path**: N/A
- **weight_status**: rule_based
- **can_detect_fire**: True
- **evaluation_status**: EVALUATED
- **note**: Heuristic scores are rule-based thresholds, NOT model accuracy.
