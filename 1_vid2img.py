import os
import cv2 


#SETTINGAN
#buat direktori untuk penyimpanan frame dulu
save_dir = "dataset/motogp/images"
os.makedirs(save_dir, exist_ok=True) 
#ambil berapa gambar tiap 1 detiknya dan nama video yang diproses
imgpersec = 1 
vid_dir = "dataset/motogp/videos"

#loop semua video didalam folder videos
for vid in os.listdir(vid_dir):
	print(vid)
	vidObj = cv2.VideoCapture(vid_dir+"/"+vid) 

	f_count = 0 #counter frame
	success = 1 #cek apakah ada frame tertangkap atau tidak, 1 = ya

	fps = round(float(vidObj.get(cv2.CAP_PROP_FPS)), 0) #bulatkan ke yang terdekat
	#print(vid+str(fps))

	count = 0
	while success: 
		success, frame = vidObj.read() #ambil status dan frame yang tertangkap
		if (f_count%(int(fps/imgpersec)) == 0):
			count += 1

		# Save image
			#penamaan image ke....
			if count >= 1000:
				scount = str(count)
			if count < 1000:
				scount = "0"+str(count)
			if count < 100:
				scount = "00"+str(count)
			if count < 10:
				scount = "000"+str(count)
			#penamaan frame
			if f_count >= 10000:
				nm = save_dir+"/"+vid+"-frame"
			if f_count < 10000:
				nm = save_dir+"/"+vid+"-frame0"
			if f_count < 1000:
				nm = save_dir+"/"+vid+"-frame00"
			if f_count < 100:
				nm = save_dir+"/"+vid+"-frame000"
			if f_count < 10:
				nm = save_dir+"/"+vid+"-frame0000"
			cv2.imwrite(nm+"%d_%s.jpg" % (f_count, scount), frame) 
			print("image: %d" % count)
		f_count += 1

	print("fps video: "+str(fps))