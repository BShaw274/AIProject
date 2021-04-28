# import the opencv library
import cv2
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
img_height = 128
img_width = 128
dice_type_model = tf.keras.models.load_model('savedModels/DiceTypeModel')
dice_roll_model = tf.keras.models.load_model('savedModels/model')

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)

    img = cv2.imread('test.jpg')
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("NewPicture.jpg",frame)
        break
image_path = '/NewPicture.jpg'
        
img = keras.preprocessing.image.load_img(
    image_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

dice_type_prediction = dice_type_model.predict(img_array)
dice_roll_prediction = dice_roll_model.predict(img_array)
type_score = tf.nn.softmax(dice_type_prediction[0])
roll_score = tf.nn.softmax(dice_roll_prediction[0])


print(
    "TYPE: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

dice_roll_prediction = dice_roll_model.predict(img_array)
roll_score = tf.nn.softmax(dice_roll_prediction[0])

print(
    "ROLL: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()