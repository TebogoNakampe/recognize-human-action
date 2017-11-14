from cnn.action_recog import run
from data.prepare import prepare, make_train_data

d = prepare()
make_train_data(d)


run()