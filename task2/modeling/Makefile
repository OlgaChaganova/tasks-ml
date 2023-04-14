CONFIG := src/configs/config.py


.PHONY: install
install:
	poetry env use 3.10
	poetry install
	poetry shell
	

.PHONY: train_recover
train_recover:
	python src/train.py --config $(CONFIG) --model_type recover


.PHONY: train_predict_noise
train_predict_noise:
	python src/train.py --config $(CONFIG) --model_type predict_noise


.PHONY: test
test:
	python src/test.py --ckpt_path $(CKPT_PATH) --data_dir $(DATA_DIR)