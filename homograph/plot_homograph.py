from plot_field import SoccerPitchConfiguration, draw_pitch
import cv2
import supervision as sv


pitch_config = SoccerPitchConfiguration()
    

pitch_image = draw_pitch(
    config=pitch_config,
    background_color=sv.Color(34, 139, 34), 
    line_color=sv.Color.WHITE,
    padding=50,
    line_thickness=4,
    point_radius=8,
    scale=0.05  
)




cv2.imshow("Football Pitch", pitch_image)


cv2.waitKey(0)
cv2.destroyAllWindows()