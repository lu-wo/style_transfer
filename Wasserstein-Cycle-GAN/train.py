import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os 
import wandb 

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

# log with wandb 
configuration = {
  "learning_rate": opt.lr,
  "batch_size": opt.batchSize,
  "lambda_A": opt.lambda_A,
  "lambda_B": opt.lambda_B,
  "image_width": opt.image_width,
  "num gen features": opt.ngf,
  "num disc features": opt.ndf,
  "model G": opt.which_model_netG,
  "model D": opt.which_model_netD,
  "image_height": opt.image_height,
  "dropout": not opt.no_dropout,
  "lambda_A": opt.lambda_A,
  "lambda_B": opt.lambda_B,
  "identity": opt.identity,
}
wandb.init(project="rey-figure-gan", entity="ds3lab-rey-figure", config=configuration)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

    # Log errors and images at epoch end
    wandb.log(dict(model.get_current_errors()))
    if epoch % 1 == 0:
        wandb.log({key: wandb.Image(img) for key, img in dict(model.get_current_visuals()).items()})
