from cnn.action_recog import run
from data.prepare import prepare, make_train_data

d = prepare()
data = make_train_data(d)


epoch = 15
batch_size = 100
run(data, batch_size, epoch)