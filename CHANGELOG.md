# Change Log

## 1.0.4

- Fix API leaking term probabilities.

## 1.0.3

- Fix index to GO term remapping

## 1.0.2

- Compensate for poorly implemented HuggingFace from_pretrained() implementation that doesn't support integer keys in the config file.

## 1.0.1

- Fix quant group size validation.

## 1.0.0

- Three separate heads for each GO aspect.
- Added adaptive aspect weighting.
- Replaced CLS vector with attention pooling.
- Configurable head depths.
