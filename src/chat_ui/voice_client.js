() => {
  let operatorSocket = null;
  let operatorReconnectDelay = 250;
  let lastEventId = 0;
  let currentSnapshot = null;
  let snapshotRefreshTimer = null;

  let voiceSocket = null;
  let voiceDesired = false;
  let voiceReconnectAttempts = 0;
  let voiceReconnectTimer = null;
  let playbackAckTimer = null;
  let stream = null;
  let context = null;
  let source = null;
  let processor = null;
  let nextPlaybackTime = 0;
  let lastLevelSentAt = 0;

  const voiceStateLabels = {
    idle: "未启动",
    connecting: "连接中",
    listening: "正在聆听",
    thinking: "Cortex 处理中",
    speaking: "正在播报（麦克风输入暂停）",
    emergency_stopped: "已紧急停止",
    error: "错误"
  };

  function wsUrl(path, query = "") {
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${window.location.host}${path}${query}`;
  }

  function setPanelText(id, text) {
    const root = document.getElementById(id);
    if (!root) return false;
    const target = root.querySelector(".prose") || root;
    target.textContent = text;
    target.style.whiteSpace = "pre-wrap";
    return true;
  }

  function renderTask(snapshot) {
    const tasks = snapshot.tasks || {};
    const active = tasks.active_task;
    const activeText = active ? `${active.status} · ${active.intent}` : "空闲";
    setPanelText(
      "operator-task-status",
      `Task 状态\n当前主任务：${activeText}\n待处理任务：${(tasks.pending_tasks || []).length}`
    );
  }

  function renderTimeline(snapshot) {
    const events = ((snapshot.tasks || {}).events || []).slice(-12).reverse();
    const lines = ["Task 时间线"];
    if (!events.length) lines.push("暂无任务事件。");
    events.forEach(event => {
      const time = (event.timestamp || "").slice(11, 19);
      lines.push(`${time}  ${event.event_type}  ${event.message || ""}`);
    });
    setPanelText("operator-task-timeline", lines.join("\n"));
  }

  function renderTelemetry(snapshot) {
    const lines = ["机器人与 Capability 状态"];
    Object.entries(snapshot.telemetry || {}).forEach(([channel, sample]) => {
      const state = sample.available === false ? "不可用" : (sample.stale ? "陈旧" : "正常");
      const age = sample.age_sec == null ? "—" : `${sample.age_sec.toFixed(1)}s 前`;
      lines.push(`${channel}: ${state} · ${age}`);
    });
    setPanelText("operator-telemetry-status", lines.join("\n"));
  }

  function renderVoice(snapshot) {
    const voice = snapshot.voice || {};
    const transcript = voice.transcript_partial || voice.transcript_final || "";
    const level = Math.round((voice.microphone_level || 0) * 100);
    const lines = [
      "实时语音",
      `Provider：${voice.provider || "unknown"}`,
      `状态：${voiceStateLabels[voice.state] || voice.state || "unknown"}`,
      `麦克风：${level}%${voice.vad_active ? " · 检测到语音" : ""}`
    ];
    if (transcript) lines.push(`转写：${transcript}`);
    if (voice.playback_pending) lines.push("播报音频仍在播放");
    if (voice.last_error) lines.push(`错误：${voice.last_error}`);
    setPanelText("operator-voice-status", lines.join("\n"));
  }

  function renderSnapshot(snapshot) {
    currentSnapshot = snapshot;
    renderTask(snapshot);
    renderTimeline(snapshot);
    renderTelemetry(snapshot);
    renderVoice(snapshot);
    if (!document.getElementById("operator-voice-status")) {
      window.setTimeout(() => currentSnapshot && renderSnapshot(currentSnapshot), 250);
    }
  }

  async function refreshSnapshot() {
    try {
      const response = await fetch("/api/operator/snapshot", { cache: "no-store" });
      if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`);
      const message = await response.json();
      lastEventId = Math.max(lastEventId, message.latest_event_id || 0);
      renderSnapshot(message.snapshot);
    } catch (error) {
      setPanelText("operator-interaction-notice", `状态刷新失败：${error.message}`);
    }
  }

  function scheduleSnapshotRefresh() {
    if (snapshotRefreshTimer) return;
    snapshotRefreshTimer = window.setTimeout(() => {
      snapshotRefreshTimer = null;
      refreshSnapshot();
    }, 80);
  }

  function applyOperatorEvent(event) {
    lastEventId = Math.max(lastEventId, event.event_id || 0);
    const payload = event.payload || {};
    if (event.kind === "voice.state" && currentSnapshot) {
      currentSnapshot.voice.state = payload.state;
      renderVoice(currentSnapshot);
    } else if (event.kind === "voice.transcript.partial" && currentSnapshot) {
      currentSnapshot.voice.transcript_partial = payload.text || "";
      renderVoice(currentSnapshot);
    } else if (event.kind === "voice.transcript.final" && currentSnapshot) {
      currentSnapshot.voice.transcript_final = payload.text || "";
      currentSnapshot.voice.transcript_partial = "";
      renderVoice(currentSnapshot);
    } else if (event.kind === "voice.vad" && currentSnapshot) {
      currentSnapshot.voice.vad_active = Boolean(payload.active);
      renderVoice(currentSnapshot);
    } else if (event.kind === "voice.microphone_level" && currentSnapshot) {
      currentSnapshot.voice.microphone_level = Number(payload.level || 0);
      renderVoice(currentSnapshot);
    } else if (event.kind.startsWith("interaction.")) {
      const detail = payload.category || payload.message || event.kind;
      setPanelText("operator-interaction-notice", `最近交互：${event.source} / ${detail}`);
      scheduleSnapshotRefresh();
    } else {
      scheduleSnapshotRefresh();
    }
  }

  function connectOperatorEvents() {
    if (operatorSocket && operatorSocket.readyState <= WebSocket.OPEN) return;
    operatorSocket = new WebSocket(
      wsUrl("/api/operator/events", `?after=${encodeURIComponent(lastEventId)}`)
    );
    operatorSocket.onopen = () => { operatorReconnectDelay = 250; };
    operatorSocket.onmessage = message => {
      const data = JSON.parse(message.data);
      if (data.type === "snapshot" || data.type === "gap") {
        lastEventId = Math.max(lastEventId, data.latest_event_id || 0);
        renderSnapshot(data.snapshot);
      } else if (data.type === "event") {
        applyOperatorEvent(data.event);
      }
    };
    operatorSocket.onclose = () => {
      operatorSocket = null;
      window.setTimeout(connectOperatorEvents, operatorReconnectDelay);
      operatorReconnectDelay = Math.min(5000, operatorReconnectDelay * 2);
    };
  }

  function resampleTo16k(input, inputRate) {
    if (inputRate === 16000) return input;
    const ratio = inputRate / 16000;
    const output = new Float32Array(Math.round(input.length / ratio));
    for (let i = 0; i < output.length; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j += 1) sum += input[j];
      output[i] = sum / Math.max(1, end - start);
    }
    return output;
  }

  function pcm16Buffer(floatSamples) {
    const buffer = new ArrayBuffer(floatSamples.length * 2);
    const view = new DataView(buffer);
    floatSamples.forEach((sample, index) => {
      const value = Math.max(-1, Math.min(1, sample));
      view.setInt16(index * 2, value < 0 ? value * 32768 : value * 32767, true);
    });
    return buffer;
  }

  function playPcm24k(arrayBuffer) {
    if (!context) return;
    const input = new Int16Array(arrayBuffer);
    const audioBuffer = context.createBuffer(1, input.length, 24000);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i += 1) channel[i] = input[i] / 32768;
    const player = context.createBufferSource();
    player.buffer = audioBuffer;
    player.connect(context.destination);
    nextPlaybackTime = Math.max(nextPlaybackTime, context.currentTime + 0.02);
    player.start(nextPlaybackTime);
    nextPlaybackTime += audioBuffer.duration;
  }

  function schedulePlaybackAck() {
    if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
    if (playbackAckTimer) window.clearTimeout(playbackAckTimer);
    const remaining = context ? Math.max(0, nextPlaybackTime - context.currentTime) : 0;
    playbackAckTimer = window.setTimeout(() => {
      playbackAckTimer = null;
      if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
        voiceSocket.send(JSON.stringify({ type: "playback.done" }));
      }
    }, Math.ceil(remaining * 1000) + 40);
  }

  function sendMicrophoneLevel(samples) {
    const now = performance.now();
    if (now - lastLevelSentAt < 200) return;
    lastLevelSentAt = now;
    let energy = 0;
    for (let i = 0; i < samples.length; i += 1) energy += samples[i] * samples[i];
    const level = Math.min(1, Math.sqrt(energy / Math.max(1, samples.length)) * 4);
    voiceSocket.send(JSON.stringify({ type: "microphone.level", level }));
  }

  function openVoiceSocket() {
    if (!voiceDesired || !stream || !context) return;
    voiceSocket = new WebSocket(wsUrl("/api/voice/stream"));
    voiceSocket.binaryType = "arraybuffer";
    voiceSocket.onopen = () => {
      voiceReconnectAttempts = 0;
      setPanelText("operator-interaction-notice", "实时语音连接已建立。");
    };
    voiceSocket.onmessage = event => {
      if (event.data instanceof ArrayBuffer) {
        playPcm24k(event.data);
        return;
      }
      const control = JSON.parse(event.data);
      if (control.type === "provider.speech_done") {
        schedulePlaybackAck();
      } else if (control.type === "provider.error" || control.type === "provider.disconnected") {
        setPanelText("operator-interaction-notice", "语音服务已断开，正在尝试重连。");
        voiceSocket.close();
      }
    };
    voiceSocket.onclose = () => {
      voiceSocket = null;
      if (!voiceDesired) return;
      if (voiceReconnectAttempts >= 3) {
        setPanelText("operator-interaction-notice", "语音连接已中断，请点击“重试语音连接”。");
        cleanupVoiceResources(false);
        voiceDesired = false;
        return;
      }
      const delay = [500, 1000, 2000][voiceReconnectAttempts];
      voiceReconnectAttempts += 1;
      setPanelText("operator-interaction-notice", `语音连接中断，${delay}ms 后重试。`);
      voiceReconnectTimer = window.setTimeout(openVoiceSocket, delay);
    };
  }

  async function prepareVoiceResources() {
    if (stream && context) return;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    context = new AudioContextClass();
    await context.resume();
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(2048, 1, 1);
    source.connect(processor);
    processor.connect(context.destination);
    processor.onaudioprocess = event => {
      if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      voiceSocket.send(pcm16Buffer(resampleTo16k(input, context.sampleRate)));
      sendMicrophoneLevel(input);
    };
  }

  function cleanupVoiceResources(closeSocket = true) {
    if (voiceReconnectTimer) window.clearTimeout(voiceReconnectTimer);
    if (playbackAckTimer) window.clearTimeout(playbackAckTimer);
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (closeSocket && voiceSocket && voiceSocket.readyState < WebSocket.CLOSING) voiceSocket.close();
    if (context) context.close();
    processor = null;
    source = null;
    stream = null;
    context = null;
    voiceSocket = null;
    voiceReconnectTimer = null;
    playbackAckTimer = null;
    nextPlaybackTime = 0;
  }

  window.ubrobotVoiceStart = async () => {
    if (voiceDesired && voiceSocket && voiceSocket.readyState <= WebSocket.OPEN) return;
    voiceDesired = true;
    voiceReconnectAttempts = 0;
    await prepareVoiceResources();
    openVoiceSocket();
  };

  window.ubrobotVoiceRetry = async () => {
    voiceDesired = false;
    cleanupVoiceResources(true);
    await window.ubrobotVoiceStart();
  };

  window.ubrobotVoiceStop = () => {
    voiceDesired = false;
    cleanupVoiceResources(true);
  };

  window.setTimeout(() => {
    refreshSnapshot();
    connectOperatorEvents();
  }, 0);
}
