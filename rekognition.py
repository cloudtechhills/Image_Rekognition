import streamlit as st
import boto3
import tempfile

# Define AWS Rekognition client
rekognition = boto3.client('rekognition')

# Define emojis
CELEBRITY_EMOJI = '🎉'
TEXT_EMOJI = '📝'
FACIAL_EMOJI = '😀'

def recognize_celebrity(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name
    uploaded_file.seek(0)  # Reset file pointer to beginning of file
    response = rekognition.recognize_celebrities(Image={'Bytes': uploaded_file.read()})
    if len(response['CelebrityFaces']) == 0:
        return 'No celebrities detected'
    else:
        celebrity = response['CelebrityFaces'][0]['Name']
        confidence = response['CelebrityFaces'][0]['MatchConfidence']
        return f'{CELEBRITY_EMOJI} {celebrity} ({confidence:.2f}%)'


def recognize_image(uploaded_file, service):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name
    uploaded_file.seek(0)
    if service == 'Celebrity':
        response = rekognition.recognize_celebrities(Image={'Bytes': uploaded_file.read()})
        if len(response['CelebrityFaces']) == 0:
            return 'No celebrities detected'
        else:
            celebrity = response['CelebrityFaces'][0]['Name']
            confidence = response['CelebrityFaces'][0]['MatchConfidence']
            return f'{CELEBRITY_EMOJI} {celebrity} ({confidence:.2f}%)'
    elif service == 'Text':
        response = rekognition.detect_text(Image={'Bytes': uploaded_file.read()})
        if len(response['TextDetections']) == 0:
            return 'No text detected'
        else:
            text = '\n'.join([detection['DetectedText'] for detection in response['TextDetections']])
            return f'{TEXT_EMOJI} {text}'
    elif service == 'Facial Analysis':
        response = rekognition.detect_faces(Image={'Bytes': uploaded_file.read()}, Attributes=['ALL'])
        if len(response['FaceDetails']) == 0:
            return 'No faces detected'
        else:
            age_range = response['FaceDetails'][0]['AgeRange']
            gender = response['FaceDetails'][0]['Gender']['Value']
            smile = response['FaceDetails'][0]['Smile']['Value']
            return f'{FACIAL_EMOJI} This person is {age_range["Low"]} to {age_range["High"]} years old, {gender}, and {"smiling" if smile else "not smiling"}'


def app():
    st.set_page_config(page_title='Recognition App', page_icon=':camera_flash:', layout='wide')

    # Define homepage
    st.markdown('# Welcome to the Recognition App!')
    st.markdown('### Upload an image and choose a recognition service to get started.')
    # st.image('https://image.freepik.com/free-vector/click-upload-file-concept-illustration_114360-3542.jpg', use_column_width=True)
    st.write('')

    # Define sidebar and main column
    options = ['Celebrity', 'Text', 'Facial Analysis']
    service = st.sidebar.selectbox('Select a recognition service', options)
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    # Define columns for image upload and recognition result
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write('')
    with col2:
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded image', use_column_width=True)
            st.write('')
            with st.spinner('Analyzing...'):
                result = recognize_image(uploaded_file, service)
                st.success(result)
    with col3:
        st.write('')

if __name__ == '__main__':
    app()
