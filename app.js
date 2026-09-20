const toast=document.getElementById('toast');
function showToast(message='Saved'){toast.textContent=message;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1600)}

document.querySelectorAll('.chart-tab').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.chart-tab').forEach(x=>x.classList.remove('active'));
    btn.classList.add('active');
    if(btn.dataset.tab!=='summary'){
      showToast(btn.textContent+' view is a demo placeholder');
    }
  })
});

document.querySelectorAll('.rail-item').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.rail-item').forEach(x=>x.classList.remove('active'));
    btn.classList.add('active');
    if(btn.dataset.view!=='schedule') showToast(btn.querySelector('em').textContent+' selected');
  })
});

document.querySelectorAll('.gap input').forEach(box=>{
  box.addEventListener('change',()=>showToast(box.checked?'Care gap marked complete':'Care gap reopened'))
});

document.querySelectorAll('button').forEach(btn=>{
  if(btn.id==='editSticky' || btn.classList.contains('chart-tab') || btn.classList.contains('rail-item')) return;
  btn.addEventListener('click',()=>{
    const label=(btn.textContent||'').trim();
    if(label) showToast(label);
  });
});

document.getElementById('editSticky').addEventListener('click',()=>{
  const text=document.getElementById('stickyText');
  const next=prompt('Update sticky note:',text.textContent);
  if(next!==null && next.trim()){
    text.textContent=next.trim();
    showToast('Sticky note updated');
  }
});

document.querySelector('.top-search input').addEventListener('keydown',e=>{
  if(e.key==='Enter'){
    showToast('Demo search: '+e.target.value);
  }
});