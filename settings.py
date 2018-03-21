AUGMENTED_VOXELS = 0

TIRE_1_CONV_OUTPUT = 5  #need to be 20

TIRE_2_CONV_OUTPUT = 5  #need to be 30

TIRE_3_CONV_OUTPUT = TIRE_2_CONV_OUTPUT

EPOCHS = 5  #need to be 30

WINDOWS_SIZE = 2

SAVING_INTERVAL = 3  #need to be 100

MEAN = 0.0

STDDEV = 0.05

FILTER_WIDTH = 3

FILTER_HEIGHT = 3

FILTER_DEPTH = 3

SMALL_FILTER_WIDTH = 1

SMALL_FILTER_HEIGHT = 1

SMALL_FILTER_DEPTH = 1

BIG_FILTER_WIDTH = 5

BIG_FILTER_HEIGHT = 5

BIG_FILTER_DEPTH = 5

CHANNELS = 1

CAD_WIDTH = 30

CAD_HEIGHT = 30

CAD_DEPTH = 30

OUTPUT_SIZE = 8 # need to be 64

BATCH_SIZE = 12

LEARNING_RATE = BATCH_SIZE * 0.0001

TARGET_ERROR_RATE = 0.001

LIMIT = 5000 # need to be 5000

FC_NEURONS = 50 # need to be 2048

COST_FUNCTION = "cross"  #cross/sqr

NUMBER_OF_TARGETS = 10

TRAIN_DAT_SET = "train_cad_10.tar.gz"

TEST_DATA_SET = "test_cad_10.tar.gz"