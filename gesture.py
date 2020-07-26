import cv2
import numpy as np
import math

camera=cv2.VideoCapture(0);

while True:

    try:
        ret,frame =camera.read();
        frame=cv2.flip(frame,1);

        filter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        cv2.rectangle( frame,(100,100),(400,400),(255,0,0), 3 ) ;                             #thumb region
        roi=frame[100:400,100:400];

        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);

        a_skin=np.array([0,48,80]);
        b_skin=np.array([20,255,255]);


        mask=cv2.inRange(hsv,a_skin,b_skin);


        mask=cv2.dilate(mask,filter, iterations=2);
        mask=cv2.erode(mask,filter,iterations=2);


        mask=cv2.GaussianBlur(mask,(3,3),0);


        contour,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE);

        areas=[cv2.contourArea(c) for c in contour]

        max_contour_index=np.argmax(areas);
        contour_max=contour[max_contour_index];

        hull = cv2.convexHull(contour_max,returnPoints=False);
        defects=cv2.convexityDefects(contour_max,hull);

        hull=cv2.convexHull(contour_max);

        n_defects=0;
        cv2.drawContours(roi,contour_max,1,(255,0,0),3)

        for i in range(defects.shape[0]) :

            area_shape = cv2.contourArea(contour_max);
            area_hull = cv2.contourArea(hull);
            solidity = area_shape / area_hull;

            start,end,far,dist=defects[i,0];

            s=tuple(contour_max[start][0]);
            e=tuple(contour_max[end][0]);
            f=tuple(contour_max[far][0]);

            a=math.sqrt((s[0]-e[0])**2 + (s[1]-e[1])**2 );
            b=math.sqrt((s[0] - f[0]) ** 2 + (s[1] - f[1]) ** 2);
            c=math.sqrt((f[0] - e[0]) ** 2 + (f[1] - e[1]) ** 2);

            angle = math.acos((b**2+c**2 - a**2 )/(2*b*c))*(180/np.pi) ;
            print(angle)

            if angle<=90:
                n_defects+=1;
                cv2.circle(roi,f,3,(0,0,255),-1);

            cv2.line(roi,s,e,(0,255,0),3);




        if solidity<=1 and solidity>=0.9 and n_defects==0:
            cv2.putText(frame, 'NULL', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3);


        elif n_defects==0 and solidity<1:

            cv2.putText(frame,'ONE',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3);

        elif n_defects==1:
            cv2.putText(frame, 'TWO', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3);

        elif n_defects==2:
            cv2.putText(frame, 'THREE', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3);

        elif n_defects==3:
            cv2.putText(frame, 'FOUR', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3);

        elif n_defects==4:
            cv2.putText(frame, 'FIVE', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3);

        else:
            cv2.putText(frame, 'REPLACE HAND', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3);




        cv2.imshow('Gesture Detection', frame);
        cv2.imshow('ROI',roi)
        cv2.imshow('MASK', mask);

    except:
        pass;



    if cv2.waitKey(1)==ord('q'):

        break;



camera.release();
cv2.destroyAllWindows()



