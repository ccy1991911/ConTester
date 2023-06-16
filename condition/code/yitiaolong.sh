python generate_artificial_sentence.py > ../data/artificial_timer.txt
cat ../data/24301_para.txt ../data/artificial_timer.txt > ../data/24301_para_n.txt
python update_condition_result_for_sentences.py
python print_condition_result.py > ../data/24301_condition.txt
cp ../data/condition_result_for_sentences.pickle ../../graph/data/
