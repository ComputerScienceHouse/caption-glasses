Captioning Heads Up Display Glasses (or CHUD Glasses) is a project started by in 2025. The goal is to create glasses that will listen to their surroundings and display what people are saying on the inside of the glasses.


## Captioning Process
Below is a flowchart of the Captioning process

```mermaid
flowchart TD
    B[Microphone] -->|Raw Audio| C(Jet Server)
    C --> E[Diarization]
    C --> F[Transcription]

    E -->|Enviornmental Sound|A
    F -->|Transcribed Text|A

    A[Capturing Device] -->|Recieved Text| G[Caption Glasses]
```

