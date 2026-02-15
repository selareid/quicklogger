function createAudioRecorder({ imageInput, recordBtn, recordingPreview, waveformPreview, updateUploadStatus }) {
  let mediaRecorder = null;
  let mediaStream = null;
  let recordedChunks = [];
  let recordedAudioBlob = null;
  let recordingPreviewUrl = null;
  let isRecording = false;

  let audioContext = null;
  let analyser = null;
  let waveformAnimationFrame = null;
  let microphoneSource = null;
  const waveformContext = waveformPreview.getContext('2d');

  function getRecordedAudioBlob() {
    return recordedAudioBlob;
  }

  function clearRecordedAudio() {
    recordedAudioBlob = null;
    clearRecordingPreview();
  }

  function setRecordingButtonState(recording) {
    isRecording = recording;

    if (recording) {
      recordBtn.textContent = '⏹';
      recordBtn.setAttribute('aria-label', 'Stop recording');
      recordBtn.title = 'Stop recording';
      recordBtn.classList.add('recording');
      return;
    }

    recordBtn.textContent = '🎤';
    recordBtn.setAttribute('aria-label', 'Start recording');
    recordBtn.title = 'Start recording';
    recordBtn.classList.remove('recording');
  }

  function clearRecordingPreview() {
    if (recordingPreviewUrl) {
      URL.revokeObjectURL(recordingPreviewUrl);
      recordingPreviewUrl = null;
    }

    recordingPreview.style.display = 'none';
    recordingPreview.removeAttribute('src');
  }

  function getRecorderOptions() {
    const preferredMimeType = 'audio/webm';
    const hasMimeTypeSupportCheck = typeof MediaRecorder.isTypeSupported === 'function';

    if (hasMimeTypeSupportCheck && MediaRecorder.isTypeSupported(preferredMimeType)) {
      return { mimeType: preferredMimeType };
    }

    return undefined;
  }

  function stopWaveformPreview() {
    if (waveformAnimationFrame) {
      cancelAnimationFrame(waveformAnimationFrame);
      waveformAnimationFrame = null;
    }

    waveformPreview.style.display = 'none';
    waveformContext.clearRect(0, 0, waveformPreview.width, waveformPreview.height);

    if (microphoneSource) {
      microphoneSource.disconnect();
      microphoneSource = null;
    }

    if (analyser) {
      analyser.disconnect();
      analyser = null;
    }

    if (audioContext) {
      audioContext.close().catch(() => { });
      audioContext = null;
    }
  }

  function startWaveformPreview(stream) {
    stopWaveformPreview();

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      return;
    }

    audioContext = new AudioContextClass();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    microphoneSource = audioContext.createMediaStreamSource(stream);
    microphoneSource.connect(analyser);

    const dataArray = new Uint8Array(analyser.fftSize);
    waveformPreview.style.display = 'block';

    const draw = () => {
      analyser.getByteTimeDomainData(dataArray);

      waveformContext.clearRect(0, 0, waveformPreview.width, waveformPreview.height);
      waveformContext.lineWidth = 2;
      waveformContext.strokeStyle = '#1a73e8';
      waveformContext.beginPath();

      const sliceWidth = waveformPreview.width / dataArray.length;
      let x = 0;

      for (let i = 0; i < dataArray.length; i++) {
        const normalized = dataArray[i] / 128.0;
        const y = (normalized * waveformPreview.height) / 2;

        if (i === 0) {
          waveformContext.moveTo(x, y);
        } else {
          waveformContext.lineTo(x, y);
        }

        x += sliceWidth;
      }

      waveformContext.lineTo(waveformPreview.width, waveformPreview.height / 2);
      waveformContext.stroke();

      waveformAnimationFrame = requestAnimationFrame(draw);
    };

    draw();
  }

  function stopStreamTracks() {
    if (!mediaStream) {
      return;
    }

    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  function onRecordingStopped() {
    const recordingMimeType = mediaRecorder.mimeType || 'audio/webm';
    recordedAudioBlob = new Blob(recordedChunks, { type: recordingMimeType });

    clearRecordingPreview();
    recordingPreviewUrl = URL.createObjectURL(recordedAudioBlob);
    recordingPreview.src = recordingPreviewUrl;
    recordingPreview.style.display = 'block';
    updateUploadStatus('Audio recording is ready to upload.');

    stopStreamTracks();
    stopWaveformPreview();
    setRecordingButtonState(false);
  }

  async function toggleRecording() {
    if (isRecording) {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
      return;
    }

    if (!window.MediaRecorder || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      updateUploadStatus('Audio recording is not supported in this browser.', true);
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      recordedAudioBlob = null;
      clearRecordingPreview();
      imageInput.value = '';

      mediaRecorder = new MediaRecorder(mediaStream, getRecorderOptions());
      mediaRecorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      });
      mediaRecorder.addEventListener('stop', onRecordingStopped);

      mediaRecorder.start();
      startWaveformPreview(mediaStream);
      setRecordingButtonState(true);
      updateUploadStatus('Recording in progress... Tap again to stop.');
    } catch (error) {
      console.error(error);
      updateUploadStatus('Unable to access your microphone.', true);
      stopWaveformPreview();
      stopStreamTracks();
      setRecordingButtonState(false);
    }
  }

  function onImageSelected() {
    if (!imageInput.files.length) {
      return;
    }

    clearRecordedAudio();

    if (isRecording && mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }

    stopWaveformPreview();
    updateUploadStatus('Image selected. Ready to upload.');
  }

  recordBtn.addEventListener('click', toggleRecording);

  return {
    getRecordedAudioBlob,
    clearRecordedAudio,
    onImageSelected,
  };
}
