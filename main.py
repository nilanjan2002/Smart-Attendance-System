import cv2 # import opencv library for image processing
import glob # import glob library for finding files with a specific pattern
import face_recognition # import face_recognition library for facial recognition tasks
import numpy as np # import numpy library for numerical operations
from datetime import datetime # import datetime library for date and time operations
from datetime import date # import date module from datetime library
import mysql.connector # import mysql connector library for connecting to a MySQL database
import os # import os library for interacting with the operating system

# Create empty lists for storing images and corresponding names
images = []
names = []

# Get current date and time
today = date.today()
now = datetime.now()

# Format the current time as a string with hours, minutes, and AM/PM indicator
dtString = now.strftime("%H:%M:%P")

# Set the path to the directory containing the training images
path = "/home/ampi/VIZION/TrainingImages/*.*"

# Iterate through all files in the training image directory
for file in glob.glob(path):
    # Read the image file and store it in the images list
    image = cv2.imread(file)
    images.append(image)

    # Get the file name and store it in the names list
    a = os.path.basename(file)
    b = os.path.splitext(a)[0]
    names.append(b)

# Define a function for encoding the training images
def encoding1(images):
    # Create an empty list for storing the encodings
    encode = []

    # Iterate through the list of images
    for img in images:
        # Get the encoding for the current image and append it to the encodings list
        unk_encoding = face_recognition.face_encodings(img)[0]
        encode.append(unk_encoding)
    
    # Return the list of encodings
    return encode

# Get the encodings for the training images
encodelist = encoding1(images)

# Define a function for adding attendance data to the MySQL database
def mysqladddata(names):
    # Connect to the MySQL database
    mydb = mysql.connector.connect(
        host="192.168.43.97",
        user="Nilanjan",
        password="itripamrit",
        database="VIZION"
    )

    # Create a cursor for executing SQL statements
    a = mydb.cursor()

    # Define the SQL insert statement
    sql = ("INSERT IGNORE INTO UserAttendance(Date,USERNAME,DATETIME) VALUE(%s,%s,%s)")

    # Set the values to insert into the database
    data = (today, names, dtString)

    # Execute the insert statement
    a.execute(sql, data)

    # Commit the changes to the database
    mydb.commit()

    # Close the connection to the database
    mydb.close()

# Initialize the video capturer
cap = cv2.VideoCapture(0)

# Initialize the video capturer
cap = cv2.VideoCapture(0)

# Continue to capture frames from the video feed until the user breaks the loop
while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame1 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(frame1)

    # Get the encodings for all the faces in the frame
    curframe_encoding = face_recognition.face_encodings(frame1, face_locations)

    # Iterate through the faces in the frame
    for encodeface, facelocation in zip(curframe_encoding, face_locations):
        # Compare the encoding of the current face to the encodings of the training images
        results = face_recognition.compare_faces(encodelist, encodeface)

        # Get the distance between the current face encoding and the encodings of the training images
        distance = face_recognition.face_distance(encodelist, encodeface)

        # Get the index of the training image with the smallest distance to the current face encoding
        match_index = np.argmin(distance)

        # Get the name of the matching training image
        name = names[match_index]

        # Add attendance data for the current face to the MySQL database
        mysqladddata(name)

        # Get the coordinates for the current face location
        x1, y1, x2, y2 = facelocation

        # Scale the coordinates for the original frame size
        x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4

        # Draw a rectangle around the face in the frame
        cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 0, 255), 3)

        # Add the name of the face to the frame
        cv2.putText(frame, name, (y2+6, x2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    # Display the frame
    cv2.imshow("FRAME", frame)

    # Break the loop if the user presses the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capturer
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
