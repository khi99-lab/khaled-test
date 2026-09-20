const sections = {
  intro:{label:"Getting started",goal:"Introduction and consent"},
  concern:{label:"Reason for visit",goal:"Presenting concern and priorities"},
  mood:{label:"Mood",goal:"Depressive symptoms and course"},
  anxiety:{label:"Anxiety",goal:"Worry, panic and somatic symptoms"},
  sleep:{label:"Sleep",goal:"Sleep quality and schedule"},
  mania:{label:"Mood elevation",goal:"Screen for mania or hypomania"},
  psychosis:{label:"Perception & thought",goal:"Screen for psychotic symptoms"},
  trauma:{label:"Trauma",goal:"Trauma exposure and current effects"},
  substances:{label:"Substances",goal:"Alcohol, nicotine and other substances"},
  treatment:{label:"Treatment history",goal:"Prior care, medication and response"},
  medical:{label:"Medical history",goal:"Medical conditions and allergies"},
  family:{label:"Family history",goal:"Psychiatric and substance history"},
  safety:{label:"Safety",goal:"Suicide, self-harm and violence screening"},
  social:{label:"Life context",goal:"Supports, work, housing and functioning"},
  goals:{label:"Goals",goal:"Patient goals and priorities"}
};

const questions = [
  {id:"consent",section:"intro",domain:"Consent",q:"Hi, I’m Dr. Adam. I’ll guide you through your intake one question at a time. You can answer in your own words, and you can skip anything you’re not ready to answer. Is it okay to begin?"},
  {id:"reason",section:"concern",domain:"Chief concern",q:"To start, what brings you in today, and what would you most like help with?"},
  {id:"onset",section:"concern",domain:"HPI",q:"When did you first notice this becoming a problem, and has it been getting better, worse, or staying about the same?"},
  {id:"function",section:"concern",domain:"Functioning",q:"How is this affecting your day-to-day life — for example work, school, relationships, parenting, or taking care of yourself?"},
  {id:"mood",section:"mood",domain:"Mood",q:"Over the last couple of weeks, how has your mood been most days?"},
  {id:"interest",section:"mood",domain:"Mood",q:"Have you noticed less interest, pleasure, motivation, or energy than usual? If yes, tell me what that has looked like for you."},
  {id:"anxiety",section:"anxiety",domain:"Anxiety",q:"How much anxiety or worry have you been dealing with lately, and what tends to trigger it?"},
  {id:"panic",section:"anxiety",domain:"Anxiety",q:"Have you had sudden episodes of intense fear or panic — things like a racing heart, shortness of breath, trembling, or feeling like something terrible is about to happen?"},
  {id:"sleep",section:"sleep",domain:"Sleep",q:"Tell me about your sleep. What time do you usually fall asleep and wake up, and are you having trouble falling asleep, staying asleep, or sleeping too much?"},
  {id:"mania",section:"mania",domain:"Mania screen",q:"Have there ever been periods when you felt unusually energized or “up,” needed much less sleep, talked much more or faster, had racing thoughts, or did things that were unusually impulsive for you?"},
  {id:"psychosis",section:"psychosis",domain:"Psychosis screen",q:"Have you ever heard or seen things other people could not, or felt that people were watching, following, or trying to harm you when others did not share that concern?"},
  {id:"trauma",section:"trauma",domain:"Trauma",q:"Have you experienced anything traumatic or deeply distressing that still affects you now? You do not need to give details you do not want to share."},
  {id:"traumaSymptoms",section:"trauma",domain:"Trauma",q:"If trauma is part of your story, do you have nightmares, unwanted memories, avoidance, feeling constantly on guard, or strong reactions to reminders?"},
  {id:"alcohol",section:"substances",domain:"Substances",q:"How often do you drink alcohol, and when you do, about how much do you typically have?"},
  {id:"substances",section:"substances",domain:"Substances",q:"Do you currently use nicotine, cannabis, prescribed controlled medications, or any other substances? Please include how often if you do."},
  {id:"pastTx",section:"treatment",domain:"Treatment",q:"Have you ever worked with a therapist, psychiatrist, psychiatric nurse practitioner, or other mental health clinician before? What was helpful or not helpful?"},
  {id:"meds",section:"treatment",domain:"Medications",q:"What medications are you taking now, including mental health medications, over-the-counter medicines, or supplements?"},
  {id:"pastMeds",section:"treatment",domain:"Medications",q:"Have you tried psychiatric medications in the past? If so, which ones helped, which did not, and did you have any side effects?"},
  {id:"medical",section:"medical",domain:"Medical",q:"Do you have any medical conditions, recent hospitalizations, seizures, head injuries, pregnancy-related concerns, or other health issues I should know about?"},
  {id:"allergies",section:"medical",domain:"Medical",q:"Do you have any medication allergies or serious medication reactions?"},
  {id:"family",section:"family",domain:"Family history",q:"Does anyone in your biological family have a history of depression, anxiety, bipolar disorder, psychosis, substance problems, suicide attempts, or psychiatric hospitalization?"},
  {id:"si",section:"safety",domain:"Safety",q:"I ask everyone this directly: recently, have you had thoughts that you would be better off dead, thoughts of hurting yourself, or thoughts of suicide?"},
  {id:"hi",section:"safety",domain:"Safety",q:"Have you had thoughts of seriously hurting someone else, or felt afraid you might lose control and hurt someone?"},
  {id:"social",section:"social",domain:"Social",q:"Who do you live with, who are the people you can rely on, and what is your current work or school situation?"},
  {id:"goals",section:"goals",domain:"Goals",q:"Last question: if treatment goes well, what changes would you most hope to notice in your life over the next few months?"}
];

const domainLabels=["Chief concern","HPI","Mood","Anxiety","Sleep","Mania screen","Psychosis screen","Trauma","Substances","Treatment","Medical","Safety"];
const state={index:0,answers:{},extra:[],finished:false,currentQuestion:null};
const chat=document.getElementById("chat"), input=document.getElementById("answerInput"), sendBtn=document.getElementById("sendBtn"), voiceStatus=document.getElementById("voiceStatus"), safetyBanner=document.getElementById("safetyBanner"), clinicianVisual=document.getElementById("clinicianVisual");
const progressBar=document.getElementById("progressBar"), progressText=document.getElementById("progressText"), sectionLabel=document.getElementById("sectionLabel"), completionPct=document.getElementById("completionPct"), currentSection=document.getElementById("currentSection"), currentGoal=document.getElementById("currentGoal"), domainGrid=document.getElementById("domainGrid"), domainCount=document.getElementById("domainCount"), snapshot=document.getElementById("snapshot");
const toast=document.getElementById("toast");
const talkModeBtn=document.getElementById("talkModeBtn"), typeModeBtn=document.getElementById("typeModeBtn"), typeComposer=document.getElementById("typeComposer"), voiceComposer=document.getElementById("voiceComposer"), liveTranscript=document.getElementById("liveTranscript"), modeHelp=document.getElementById("modeHelp");
let recognition=null, recognizing=false, voiceUnlocked=false, sessionActive=false, isSpeaking=false, silenceTimer=null, pendingAutoSubmit=false, accumulatedTranscript="", answerMode="type";

function showToast(msg){toast.textContent=msg;toast.classList.add("show");setTimeout(()=>toast.classList.remove("show"),1500)}
function escapeText(v){return String(v||"").replace(/[<>]/g,"")}
function addDoctor(text){const t=document.getElementById("doctorMessageTemplate").content.cloneNode(true);t.querySelector("p").textContent=text;chat.appendChild(t);scrollChat()}
function addPatient(text){const t=document.getElementById("patientMessageTemplate").content.cloneNode(true);t.querySelector("p").textContent=text;chat.appendChild(t);scrollChat()}
function addSystem(text){const d=document.createElement("div");d.className="message system-message";d.innerHTML='<div class="bubble">'+escapeText(text)+'</div>';chat.appendChild(d);scrollChat()}
function scrollChat(){requestAnimationFrame(()=>chat.scrollTo({top:chat.scrollHeight,behavior:"smooth"}))}

function transitionFor(question){
  if(!document.getElementById("naturalTransitions").checked || state.index===0) return "";
  const phrases={
    mood:"Thanks for explaining that. I want to understand your mood a little better. ",
    anxiety:"That helps. Now I’d like to ask about anxiety and worry. ",
    sleep:"I appreciate that. Let’s talk about sleep for a moment. ",
    mania:"I ask everyone a few questions about changes in energy and mood. ",
    psychosis:"I’m going to ask another routine screening question. ",
    trauma:"Thank you. I’m going to shift to trauma, and you can keep this as general as you want. ",
    substances:"Next I’ll ask about substances because they can affect sleep, mood, and medications. ",
    treatment:"Now I’d like to understand what treatment you’ve already tried. ",
    medical:"A few medical questions next. ",
    family:"I’m going to ask briefly about family history. ",
    safety:"Thank you for staying with me. I ask every patient these safety questions directly. ",
    social:"We’re almost finished. I’d like to understand your support system and daily life. ",
    goals:"One last thing. "
  };
  const prev=state.currentQuestion?.section;
  return prev!==question.section ? (phrases[question.section]||"") : "";
}

function getQuestion(){
  if(state.extra.length) return state.extra.shift();
  return questions[state.index] || null;
}

function askNext(){
  const q=getQuestion();
  if(!q){finish();return}
  state.currentQuestion=q;
  const prefix=transitionFor(q);
  const text=prefix+q.q;
  addDoctor(text);
  updateUI();
  if(answerMode==="talk" && voiceUnlocked) speak(text);
}

function selectBritishMaleVoice(){
  const voices=speechSynthesis.getVoices();
  const ranked=[
    /Daniel/i,/Arthur/i,/Oliver/i,/Ryan/i,/George/i,/Google UK English Male/i,/Microsoft.*(Ryan|George|Thomas|Oliver)/i
  ];
  for(const pattern of ranked){
    const found=voices.find(v=>pattern.test(v.name) && /^en-GB/i.test(v.lang||""));
    if(found)return found;
  }
  return voices.find(v=>/^en-GB/i.test(v.lang||"") && !/female|samantha|serena|karen/i.test(v.name))
      || voices.find(v=>/^en-GB/i.test(v.lang||""))
      || voices.find(v=>/^en/i.test(v.lang||""));
}
function speak(text){
  if(!("speechSynthesis" in window)){showToast("Voice playback is not supported in this browser");return}
  stopListening();
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  u.rate=.94;u.pitch=.92;u.lang="en-GB";
  const preferred=selectBritishMaleVoice();
  if(preferred)u.voice=preferred;
  u.onstart=()=>{
    isSpeaking=true;
    clinicianVisual.classList.add("speaking");
    voiceStatus.textContent="Dr. Adam is speaking…";
  };
  u.onend=()=>{
    isSpeaking=false;
    clinicianVisual.classList.remove("speaking");
    voiceStatus.textContent=sessionActive ? "Your turn — just speak. I’m listening automatically." : "Your turn — answer by voice or typing.";
    if(sessionActive) setTimeout(startListening,350);
  };
  u.onerror=()=>{
    isSpeaking=false;
    clinicianVisual.classList.remove("speaking");
    voiceStatus.textContent="Audio could not play. Tap Repeat to try again.";
    if(sessionActive) setTimeout(startListening,350);
  };
  speechSynthesis.speak(u);
}

function classifyRisk(text){
  const t=text.toLowerCase();
  const concerning=["kill myself","suicide","suicidal","end my life","hurt myself","self harm","self-harm","better off dead","hurt someone","kill someone"];
  const negated=["no suicidal","not suicidal","never suicidal","no thoughts","don't want to hurt","do not want to hurt","denies"];
  return concerning.some(k=>t.includes(k)) && !negated.some(k=>t.includes(k));
}

function maybeAddFollowup(q,text){
  if(!document.getElementById("adaptiveFollowups").checked) return;
  const t=text.toLowerCase();
  if(q.id==="si" && classifyRisk(text)){
    state.extra.push({id:"siFollow",section:"safety",domain:"Safety",q:"Thank you for telling me. Are you having those thoughts right now, and do you have a specific plan, intent, or access to anything you might use to hurt yourself?",followup:true});
  }
  if(q.id==="mania" && /yes|sometimes|have been|definitely|often/.test(t) && !/no|never/.test(t)){
    state.extra.push({id:"maniaFollow",section:"mania",domain:"Mania screen",q:"During those periods, about how long did the change last, and did other people notice that you were different from your usual self?",followup:true});
  }
  if(q.id==="psychosis" && /yes|sometimes|voices|hearing|seeing|watching|following/.test(t) && !/no|never/.test(t)){
    state.extra.push({id:"psychosisFollow",section:"psychosis",domain:"Psychosis screen",q:"When that happens, how often does it occur, and how convinced do you feel in the moment that the experience is real?",followup:true});
  }
  if(q.id==="panic" && /yes|panic|attack/.test(t) && !/no|never/.test(t)){
    state.extra.push({id:"panicFollow",section:"anxiety",domain:"Anxiety",q:"How often are those panic episodes happening, about how long do they last, and have you started avoiding anything because you fear another one?",followup:true});
  }
}

function submitAnswer(skipped=false){
  if(state.finished)return;
  const q=state.currentQuestion;
  if(!q)return;
  clearSilenceTimer();
  stopListening();
  pendingAutoSubmit=false;
  const text=skipped?"Skipped":input.value.trim();
  if(!text)return;
  addPatient(text);
  state.answers[q.id]={question:q.q,answer:text,section:q.section,domain:q.domain,followup:!!q.followup};
  if(classifyRisk(text)){safetyBanner.classList.remove("hidden");}
  maybeAddFollowup(q,text);
  input.value="";
  accumulatedTranscript="";
  liveTranscript.textContent="Your answer will appear here while you speak.";
  if(!q.followup) state.index++;
  updateUI();
  setTimeout(askNext,350);
}

function updateUI(){
  const q=state.currentQuestion||questions[Math.min(state.index,questions.length-1)];
  const sec=sections[q?.section]||sections.intro;
  sectionLabel.textContent=sec.label;
  currentSection.textContent=sec.label;
  currentGoal.textContent=sec.goal;
  const answeredCore=Math.min(state.index,questions.length);
  const pct=Math.round((answeredCore/questions.length)*100);
  progressBar.style.width=pct+"%";
  completionPct.textContent=pct+"%";
  progressText.textContent=Math.min(state.index+1,questions.length)+" of "+questions.length;
  document.querySelector(".progress-wrap").style.display=document.getElementById("showProgress").checked?"block":"none";
  renderDomains();renderSnapshot();
}

function renderDomains(){
  const done=new Set(Object.values(state.answers).filter(x=>x.answer!=="Skipped").map(x=>x.domain));
  domainGrid.innerHTML="";
  domainLabels.forEach(label=>{
    const el=document.createElement("div");
    el.className="domain"+(done.has(label)?" done":"");
    el.innerHTML="<i></i><span>"+label+"</span>";
    domainGrid.appendChild(el);
  });
  domainCount.textContent=domainLabels.filter(d=>done.has(d)).length+"/"+domainLabels.length+" captured";
}

function renderSnapshot(){
  const entries=Object.entries(state.answers).filter(([k,v])=>v.answer!=="Skipped");
  if(!entries.length){snapshot.className="snapshot empty";snapshot.textContent="Responses will be organized here as the conversation progresses.";return}
  snapshot.className="snapshot";
  const latest=entries.slice(-8);
  snapshot.innerHTML=latest.map(([id,v])=>'<div class="snapshot-item"><strong>'+escapeText(v.domain)+'</strong>'+escapeText(v.answer)+'</div>').join("");
}

function buildSummary(){
  const grouped={};
  Object.values(state.answers).forEach(v=>{
    if(v.answer==="Skipped")return;
    (grouped[v.domain] ||= []).push(v.answer);
  });
  return Object.entries(grouped).map(([domain,vals])=>domain+": "+vals.join(" | ")).join("\n");
}

function finish(){
  state.finished=true;
  progressBar.style.width="100%";completionPct.textContent="100%";progressText.textContent=questions.length+" of "+questions.length;
  addDoctor("Thank you. That completes the intake conversation. Your answers are now organized for clinician review.");
  const card=document.createElement("div");card.className="finish-card";
  card.innerHTML='<h3>Intake complete</h3><p>This prototype captured the conversation into structured clinical domains. A production version can send the structured data to the clinician workspace for review before anything is signed or acted on.</p><div class="finish-actions"><button id="reviewSummary" class="primary">View captured summary</button><button id="startOverInline">Start over</button></div>';
  chat.appendChild(card);scrollChat();
  card.querySelector("#reviewSummary").onclick=()=>{const s=buildSummary();alert(s||"No structured responses captured.");};
  card.querySelector("#startOverInline").onclick=restart;
  input.disabled=true;sendBtn.disabled=true;talkModeBtn.disabled=true;typeModeBtn.disabled=true;
}

function restart(){
  speechSynthesis?.cancel?.();
  sessionActive=false;pendingAutoSubmit=false;isSpeaking=false;clearSilenceTimer();stopListening();answerMode="type";typeModeBtn.classList.add("active");talkModeBtn.classList.remove("active");typeComposer.classList.remove("hidden");voiceComposer.classList.add("hidden");
  state.index=0;state.answers={};state.extra=[];state.finished=false;state.currentQuestion=null;
  chat.innerHTML="";safetyBanner.classList.add("hidden");input.disabled=false;sendBtn.disabled=false;talkModeBtn.disabled=false;typeModeBtn.disabled=false;input.value="";liveTranscript.textContent="Your answer will appear here while you speak.";
  updateUI();askNext();
}

function clearSilenceTimer(){
  if(silenceTimer){clearTimeout(silenceTimer);silenceTimer=null}
}

function armSilenceTimer(){
  clearSilenceTimer();
  if(!sessionActive)return;
  silenceTimer=setTimeout(()=>{
    if(!recognizing || !input.value.trim())return;
    pendingAutoSubmit=true;
    voiceStatus.textContent="Got it — moving to the next question…";
    stopListening();
  },2100);
}

function startListening(){
  if(!recognition || answerMode!=="talk" || !sessionActive || state.finished || isSpeaking || recognizing)return;
  pendingAutoSubmit=false;
  accumulatedTranscript="";
  try{recognition.start()}catch(e){}
}

function stopListening(){
  clearSilenceTimer();
  if(recognition && recognizing){
    try{recognition.stop()}catch(e){}
  }
}

function initRecognition(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){
    voiceStatus.textContent="Hands-free voice input is not available in this browser. Typing still works.";
    talkModeBtn.disabled=true;
    return;
  }
  recognition=new SR();
  recognition.lang="en-US";
  recognition.interimResults=true;
  recognition.continuous=true;

  recognition.onstart=()=>{
    recognizing=true;
    voiceComposer.classList.add("listening");
    voiceStatus.textContent=sessionActive ? "Listening automatically… speak naturally." : "Listening… speak naturally.";
  };

  recognition.onresult=e=>{
    let interim="";
    for(let i=e.resultIndex;i<e.results.length;i++){
      const tr=e.results[i][0].transcript.trim();
      if(e.results[i].isFinal){
        accumulatedTranscript=(accumulatedTranscript+" "+tr).trim();
      }else{
        interim=(interim+" "+tr).trim();
      }
    }
    input.value=(accumulatedTranscript+" "+interim).trim();
    liveTranscript.textContent=input.value.trim() || "Listening…";
    if(input.value.trim()) armSilenceTimer();
  };

  recognition.onerror=e=>{
    recognizing=false;
    voiceComposer.classList.remove("listening");
    if(e.error==="not-allowed" || e.error==="service-not-allowed"){
      sessionActive=false;
      setAnswerMode("type", false);
      voiceStatus.textContent="Microphone permission is required for hands-free mode.";
      return;
    }
    if(sessionActive && !isSpeaking && !state.finished){
      voiceStatus.textContent="Reconnecting microphone…";
      setTimeout(startListening,500);
    }else{
      voiceStatus.textContent=answerMode==="talk"?"Voice input paused; reconnecting…":"Type your answer and press Send.";
    }
  };

  recognition.onend=()=>{
    recognizing=false;
    voiceComposer.classList.remove("listening");
    clearSilenceTimer();

    if(pendingAutoSubmit && input.value.trim()){
      pendingAutoSubmit=false;
      setTimeout(()=>submitAnswer(false),120);
      return;
    }

    if(sessionActive && !isSpeaking && !state.finished){
      voiceStatus.textContent="Listening automatically…";
      setTimeout(startListening,350);
    }else if(!state.finished){
      voiceStatus.textContent=answerMode==="talk"?"Listening automatically…":"Type your answer and press Send.";
      input.focus();
    }
  };
}

sendBtn.addEventListener("click",()=>submitAnswer(false));
document.getElementById("skipBtn").addEventListener("click",()=>submitAnswer(true));
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();submitAnswer(false)}});
function setAnswerMode(mode, replayQuestion=true){
  answerMode=mode;
  const talk=mode==="talk";
  talkModeBtn.classList.toggle("active",talk);
  typeModeBtn.classList.toggle("active",!talk);
  typeComposer.classList.toggle("hidden",talk);
  voiceComposer.classList.toggle("hidden",!talk);

  if(talk){
    voiceUnlocked=true;
    sessionActive=true;
    modeHelp.textContent="Talk mode is hands-free: Dr. Adam speaks, then listens, and your answer sends automatically after a short pause.";
    voiceStatus.textContent="Talk mode is on.";
    if(replayQuestion && state.currentQuestion){
      speak((transitionFor(state.currentQuestion)||"")+state.currentQuestion.q);
    }else if(!isSpeaking){
      setTimeout(startListening,250);
    }
  }else{
    sessionActive=false;
    pendingAutoSubmit=false;
    clearSilenceTimer();
    stopListening();
    speechSynthesis?.cancel?.();
    isSpeaking=false;
    clinicianVisual.classList.remove("speaking");
    voiceComposer.classList.remove("listening");
    modeHelp.textContent="Type your answer and press Send, or choose Talk for a fully hands-free conversation.";
    input.focus();
  }
}

talkModeBtn.addEventListener("click",()=>setAnswerMode("talk",true));
typeModeBtn.addEventListener("click",()=>setAnswerMode("type",false));
document.getElementById("repeatBtn").addEventListener("click",()=>{voiceUnlocked=true;if(state.currentQuestion)speak((transitionFor(state.currentQuestion)||"")+state.currentQuestion.q)});
document.getElementById("restartBtn").addEventListener("click",restart);
document.getElementById("copySummaryBtn").addEventListener("click",async()=>{const s=buildSummary();if(!s){showToast("No responses yet");return}try{await navigator.clipboard.writeText(s);showToast("Summary copied")}catch{showToast("Copy unavailable")}});
document.getElementById("showProgress").addEventListener("change",updateUI);


initRecognition();setAnswerMode("type",false);renderDomains();updateUI();askNext();
