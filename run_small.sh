#python main.py --cuda --log-interval 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    # Test perplexity of 75.96
#python main.py --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        # Test perplexity of 77.42
#python main.py --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied # Test perplexity of 72.30
python main.py --gpuid 0 --data ./data/wiki --cuda --log-interval 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 20  --save small.pt         # Test perplexity of 80.97
python main.py --gpuid 0 --data ./data/wiki --cuda --log-interval 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 20  --tied --save small-t.pt         # Test perplexity of 80.97
