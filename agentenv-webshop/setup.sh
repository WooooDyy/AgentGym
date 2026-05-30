cd ./webshop
pip install -r requirements.txt
pip install -U "Werkzeug>=2,<2.3" "mkl>=2021,<2022" "typing_extensions<4.6.0" "gym==0.23.1"
pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl#sha256=6ce19d37dfe5280400f80a5954d41afca10cbc742b97bfcf4b0e452b6eb24273
# python -m spacy download en_core_web_lg


cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
bash ./run_indexing.sh
cd ../..

pip install -e .

pip uninstall numpy

pip install numpy
