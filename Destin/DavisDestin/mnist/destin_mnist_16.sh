#bin/bash

../bin/destin train config_16 data/train_out.bin data/train.bin data/labels.bin 1
../bin/destin test config_16 data/train_out.bin data/train.bin data/labels.bin 1
