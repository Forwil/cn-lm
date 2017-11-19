#python main.py --cuda --log-interval 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    # Test perplexity of 75.96
#python main.py --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        # Test perplexity of 77.42
#python main.py --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied # Test perplexity of 72.30
#python main.py --gpuid 1 --data ./data/wiki --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.5 --epochs 20 --save big.pt         # Test perplexity of 80.
python main.py --gpuid 1 --data ../data/all_and_book --cuda --log-interval 1 --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 10 --tied --save all_book_big.pt  --flag all_book       # Test perplexity of 80.
