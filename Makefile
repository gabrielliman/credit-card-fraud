COMP := playground-series-s6e11
PY := conda run --no-capture-output -n fraud python
CMP := conda run --no-capture-output -n fraud python -m src.compare
.PHONY: install data eda run submit compare
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
compare:        # make compare A=experiments/exp02_lgbm_default B=experiments/exp03_lgbm_sized
	$(CMP) $(A) $(B)
