[monitoring.py]

elif selected == "denseModel":
    col2_box_msg = fallpredict.is_fallenlearned("dense", models["dense"], convert_landmarks_to_row(frame_landmarks))
    fall_msg = "dense 모델을 통한 낙상판단입니다."
elif selected == "lstmModel":
    col2_box_msg = fallpredict.is_fallenlearned("lstm", models["lstm"], convert_landmarks_to_row(frame_landmarks))
    fall_msg = "lstm 모델을 통한 낙상판단입니다."
elif selected == "ensembleModel":
    col2_box_msg = fallpredict.is_fallenlearned("ensemble", models["ensemble"], convert_landmarks_to_row(frame_landmarks))
    fall_msg = "ensemble 모델을 통한 낙상판단입니다."
else :
    col2_box_msg("모델로드 오류입니다.")