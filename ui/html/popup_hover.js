(function() {
  var timers = {};
  document.querySelectorAll('.lbl').forEach(function(el) {
    var popup = el.querySelector('.popup');
    if (!popup) return;
    var id = el.dataset.nid;
    el.addEventListener('mouseenter', function() {
      timers[id] = setTimeout(function() { popup.style.display = 'block'; }, 200);
    });
    el.addEventListener('mouseleave', function() {
      clearTimeout(timers[id]);
      popup.style.display = 'none';
    });
    popup.addEventListener('mouseenter', function() { clearTimeout(timers[id]); });
    popup.addEventListener('mouseleave', function() { popup.style.display = 'none'; });
  });
})();

