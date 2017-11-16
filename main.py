from cnn.action_recog import run
from data.prepare import prepare, make_train_data

size = [80, 60]

d = prepare(size)
data = make_train_data(d)


epoch = 20
batch_size = 100
run(data, batch_size, epoch)