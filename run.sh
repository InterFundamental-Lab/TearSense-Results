rm -rf outputs/ external_assessor/ __pycache__/

cp -r ../tearsense_results/* ./

source ~/.zshrc
conda activate tear_assessor

python run.py
