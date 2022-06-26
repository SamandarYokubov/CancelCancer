from config import Config
import logging
from aiogram import Bot, Dispatcher, executor, types
from matplotlib import pyplot as plt
import numpy as np
import cv2
from models import UNet
from models import ClassificationModel
import torch
from matplotlib import rcParams
import tensorflow as tf
from skimage.transform import resize
rcParams['figure.figsize'] = (15,4)


def predict(model, data):
    model.eval()
    with torch.set_grad_enabled(False):
        Y_pred = model(data.to(device))
        Y_pred[Y_pred < 0.4] = 0
        Y_pred[Y_pred > 0.4] = Y_pred[Y_pred > 0.4] * 2
    return Y_pred


cfg = Config()

# creating segmentation model
device = torch.device('cpu')

print(device)
seg_model = UNet().to(device)
seg_model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH_SEGMENTATION))
seg_model.eval()


# creating classification model
with tf. device("cpu:0"):
    clf_model = ClassificationModel().get_model()
    clf_model = tf.keras.models.load_model(cfg.CHECKPOINT_PATH_CLASSIFICATION)



# creating bot dispatcher object
bot = Bot(token=cfg.BOT_TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)



# start command
@dp.message_handler(commands=["start"])
async def cmd_test1(message: types.Message):
    await message.reply("I am CancelCancer")


def classify_image(image):
    size = (180,180)
    image = resize(image, size, mode='constant', anti_aliasing=True)
    print(image.shape)    
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    image = tf.convert_to_tensor(image)
    with tf. device("cpu:0"):
        results = clf_model.predict(image)
        predictions = tf.nn.softmax(results)
        predictions = predictions.numpy()
        print(predictions)
        predictions_id = np.argmax(predictions)
        if predictions[0][predictions_id] < 0.5:
            return 'Всё в порядке!'
        predicted_class_name = cfg.out_class[predictions_id]
    return predicted_class_name


def segment_image(image, output_file_path):
    size = (256,256)
    h, w, c = image.shape
    image = resize(image, size, mode='constant', anti_aliasing=True)
    print(image.shape)    
    image = np.expand_dims(image, axis=0)
    image = np.rollaxis(image, 3, 1)
    print(image.shape)
    image = torch.from_numpy(image).to(torch.float32)
    #results = seg_model(image.to(device))
    results = predict(seg_model, image)    
    print(results, results.sum(), results.max(), results.mean())
    output_image = np.squeeze(results.cpu().detach().numpy())
    output_image = resize(output_image, (h,w),  mode='constant', anti_aliasing=True)
    plt.imshow(output_image, cmap='gray')
    plt.savefig(output_file_path, bbox_inches="tight")
    

# handling photo
@dp.message_handler(content_types=["photo"])
async def download_photo(message: types.Message):
    user_id = message.from_user.id
    save_file_path_in = f"./photos/{user_id}_in.bmp"
    save_file_name_out = f"./photos/model_{user_id}_out.jpg"
    await message.photo[-1].download(destination_file=save_file_path_in)    
    image = cv2.imread(save_file_path_in, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cancer_class = classify_image(image)
    segment_image(image, save_file_name_out)
    message_text = cancer_class
    await bot.send_message(message.from_user.id, message_text)
    await bot.send_photo(chat_id = message.from_user.id,
                        photo = open(save_file_name_out, "rb"),
                        reply_to_message_id=message.message_id)



if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)