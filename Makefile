COMP := playground-series-s6e11
PY := conda run --no-capture-output -n fraud python
.PHONY: install data eda run submit
install:
	conda run -n fraud pip install -r requirements.txt
data:
	conda run -n fraud kaggle competitions download -c $(COMP) -p data && unzip -o data/$(COMP).zip -d data && rm data/$(COMP).zip
eda:
	$(PY) -m src.eda_profile data
run:            # make run EXP=experiments/exp02_lgbm_default
	$(PY) -m src.run $(EXP)
submit:         # make submit EXP=experiments/exp02_lgbm_default
	conda run -n fraud kaggle competitions submit -c $(COMP) -f $(EXP)/submission.csv -m "$$(basename $(EXP))"
