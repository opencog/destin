#bin/bash

../bin/destin train config_32 data/destin_train_32_out.bin data/destin_train_32.bin data/labels.bin 0
../bin/destin test config_32 data/destin_train_32_out.bin data/destin_train_32.bin data/labels.bin 0

../bin/destin train config_32 data/destin_train_nn_32_out.bin data/destin_train_nn_32.bin data/labels.bin 0
../bin/destin test config_32 data/destin_train_nn_32_out.bin data/destin_train_nn_32.bin data/labels.bin 0
