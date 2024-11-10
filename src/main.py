import sys
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger
from laminar import laminar_main_text, laminar_assistant 
from fastapi import Form


sys.path.append("/app/")
from src.rnnt_inference import process_audio_file, load_model

app = FastAPI()

model = load_model()

@app.get("/")
async def root():
    """Root endpoint to confirm server status."""
    return {"message": "Server is running"}

async def process_audio_request(file: UploadFile):
    """Processes the uploaded audio file for transcription."""
    temp_file_path = "/app/audio"
    extension = file.filename.split(".")[-1]

    if extension not in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        logger.error(f"Unsupported file format: {extension}")
        raise HTTPException(status_code=400, detail="Unsupported file format")

    original_file_path = f"{temp_file_path}_original.{extension}"
    with open(original_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcription = process_audio_file(original_file_path, model)

    if transcription in ["Transcription failed", "Audio processing failed"]:
        logger.error("Audio processing failed during transcription.")
        return {"transcription": "something wrong with transcription of file"}

    return {"transcription": transcription[0]}


@app.post("/process-main-text/")
async def process_main_text(file: UploadFile = File(...)):
    """Transcribes the audio file and processes the transcribed text using Laminar's main text pipeline."""
    logger.info("Processing audio for main text.")
    
    transcription_response = await process_audio_request(file)
    
    if transcription_response.get("transcription") == "something wrong with transcription of file":
        logger.error("Transcription failed.")
        return transcription_response

    transcribed_text = transcription_response["transcription"]
    logger.info(f"Transcription successful: {transcribed_text}")
    
    laminar_output = laminar_main_text({"text": transcribed_text})
    processed_text = laminar_output.outputs['Output_text']['value'].replace('\n', '')
    
    logger.info(f"Processed text from Laminar: {processed_text}")
    return {"processed_text": processed_text}

@app.post("/process-with-assistant/")
async def process_with_assistant(text: str = Form(...), file: UploadFile = File(...)):

    """Transcribes the audio file for instruction and processes it along with the provided text using Laminar's assistant pipeline."""
    logger.info("Processing audio for instruction with assistant.")
    
    transcription_response = await process_audio_request(file)
    
    if transcription_response.get("transcription") == "something wrong with transcription of file":
        logger.error("Transcription failed.")
        return transcription_response

    instruction = transcription_response["transcription"]
    logger.info(f"Instruction transcribed: {instruction}")
    
    laminar_output = laminar_assistant({"text": text, "instruction": instruction})
    updated_text = laminar_output.outputs['output']['value']
    
    logger.info(f"Updated text from Laminar: {updated_text}")
    return {"updated_text": updated_text}
