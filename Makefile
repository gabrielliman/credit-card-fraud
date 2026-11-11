COMP := playground-series-s6e11
.PHONY: install data train submit
install:
	pip install -r requirements.txt
data:
	kaggle competitions download -c $(COMP) -p data && unzip -o data/$(COMP).zip -d data && rm data/$(COMP).zip
train:
	python -m src.train
submit:
	kaggle competitions submit -c $(COMP) -f $(FILE) -m "$(MSG)"
