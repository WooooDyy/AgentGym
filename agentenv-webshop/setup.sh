cd ./webshop
pip install -r requirements.txt
pip install -U "Werkzeug>=2,<3" "mkl>=2021,<2022" "typing_extensions<4.6.0" "gym==0.23.1"
python -m spacy download en_core_web_lg

cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
bash ./run_indexing.sh
cd ../..
pip install -e .
