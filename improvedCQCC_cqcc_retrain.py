"""Final Akıncı–Özbay CQCC model training (same exact CV-selected epoch procedure)."""
import sys
from improvedCQCC_cqcc_train import main
if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--stage', 'final', *sys.argv[1:]]
    main()
