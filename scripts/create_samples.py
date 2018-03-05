# create_samples.py
import cv2, sys, numpy, os
size = 2
count = 0
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
try:
    img_path = sys.argv[1]
except:
    print("Provide location of subject images ie;/home/path/to/images")
    sys.exit(0)
sub_name=os.path.basename(img_path)
path = os.path.join(fn_dir, sub_name)
if not os.path.isdir(path):
    os.mkdir(path)

images=os.listdir(img_path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)

# Beginning message
print("\n\033[94m Creating training samples from images in " +img_path+ "\033[0m\n")

for image in images:
	# Open the image
	img = cv2.imread(os.path.join(img_path,image),1)
	# Get image size
    	height, width, channels = img.shape
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Scale down for speed
	mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
	# Detect faces
    	faces = haar_cascade.detectMultiScale(mini)
	# We only consider largest face
    	faces = sorted(faces, key=lambda x: x[3])
	if(len(faces)>1):
		print("\n\033[91m Multiple faces detected in " +image+ "\033[0m\n")
		print("\n\033[91m For best results use images with one face only \033[0m\n")
	if faces:
		face_i = faces[0]
		(x, y, w, h) = [v * size for v in face_i]
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (im_width, im_height))
		if(w * 6 < width or h * 6 < height):
			print("Face too small in "+image)
		else:

			# Save image file
			count+=1
			cv2.imwrite('%s/%s.png' % (path, count), face_resize)
			

print("\n\033[92m "+str(count)+" samples created from "+str(len(images))+" images in "+img_path+"\033[0m\n")
