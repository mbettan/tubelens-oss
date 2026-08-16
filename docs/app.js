// TubeLens OSS — Interactive Documentation & Agent Demo JS
document.addEventListener('DOMContentLoaded', () => {
  // Clean up any stale light-theme state from prior versions
  try { localStorage.removeItem('theme'); } catch (e) {}
  document.documentElement.removeAttribute('data-theme');

  // 2. Mobile Navigation Menu Toggle
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const navLinks = document.querySelector('.nav-links');

  if (mobileMenuBtn && navLinks) {
    mobileMenuBtn.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      mobileMenuBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });

    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
        mobileMenuBtn.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // 3. Config Tab Switcher
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const activeContent = document.getElementById(targetId);
      if (activeContent) activeContent.classList.add('active');
    });
  });

  // 4. Copy Code Buttons with Fallback & Feedback
  async function copyTextToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for non-HTTPS or legacy environments
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand('copy');
      textArea.remove();
      return Promise.resolve();
    } catch (err) {
      textArea.remove();
      return Promise.reject(err);
    }
  }

  const copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const codeId = btn.getAttribute('data-code');
      const codeEl = document.getElementById(codeId);
      if (codeEl) {
        try {
          await copyTextToClipboard(codeEl.innerText.trim());
          const origText = btn.innerText;
          btn.innerText = 'Copied!';
          btn.style.color = '#10b981';
          btn.style.borderColor = '#10b981';
          setTimeout(() => {
            btn.innerText = origText;
            btn.style.color = '';
            btn.style.borderColor = '';
          }, 2000);
        } catch (err) {
          console.error('Clipboard copy error:', err);
        }
      }
    });
  });

  // 5. Agent Demo — Animated MCP Workflow
  const agentChat = document.getElementById('agent-chat');
  const progressBar = document.getElementById('agent-progress-bar');
  const progressLabel = document.getElementById('agent-progress-label');
  const replayBtn = document.getElementById('agent-replay-btn');

  if (!agentChat) return;

  // Escape HTML
  function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  // Demo scenario: Full agent research workflow
  const steps = [
    {
      type: 'user',
      text: 'Research @mbettan on YouTube. What does he recommend about building leverage as a solopreneur?',
      progress: 5,
      label: 'User prompt received'
    },
    {
      type: 'thinking',
      text: 'I\'ll start by resolving the channel handle to get the channel ID...',
      progress: 10,
      label: 'Planning tool calls'
    },
    {
      type: 'tool',
      name: 'youtube_resolve_channel',
      params: { query: '@mbettan' },
      response: {
        label: '✓ Channel resolved',
        items: [
          { icon: '📺', text: 'Michael Bettan' },
          { icon: '🔑', text: 'UCx7K_pFp3a2CnH6GVvHJXdQ' },
          { icon: '👥', text: '412 subscribers · 31 videos' }
        ]
      },
      progress: 20,
      label: 'Channel resolved'
    },
    {
      type: 'thinking',
      text: 'Found the channel. Let me browse recent videos to find leverage-related content...',
      progress: 25,
      label: 'Browsing catalog'
    },
    {
      type: 'tool',
      name: 'youtube_list_channel_videos',
      params: { channel_id: 'UCx7K_pFp3a2CnH6GVvHJXdQ', query: 'leverage', limit: 5, sort: 'newest' },
      response: {
        label: '✓ 3 matching videos found',
        items: [
          { icon: '🎬', text: 'The Leverage Hierarchy — How to Scale Without Hiring (31 views · 30 min)' },
          { icon: '🎬', text: 'AI Agents as Leverage: Replacing Teams with Systems (89 views · 22 min)' },
          { icon: '🎬', text: 'Why I Rejected $40K/mo Revenue (12 views · 18 min)' }
        ]
      },
      progress: 35,
      label: 'Videos found'
    },
    {
      type: 'thinking',
      text: 'The first video looks most relevant. Let me analyze it deeply...',
      progress: 40,
      label: 'Selecting video for analysis'
    },
    {
      type: 'tool',
      name: 'youtube_analyze_video',
      params: { video_url: 'https://youtube.com/watch?v=HmQkMRZFd8s' },
      response: {
        label: '✓ Analysis complete (30 min video processed)',
        items: [
          { icon: '📝', text: 'Summary: Argues for a 4-tier leverage hierarchy — code, media, capital, labor — and why solopreneurs should reject revenue requiring headcount.' },
          { icon: '🏷️', text: 'Topics: leverage hierarchy, lump of labor fallacy, AI agents, global talent, revenue rejection' },
          { icon: '📎', text: '6 timestamped topic sections indexed' }
        ]
      },
      progress: 55,
      label: 'Video analyzed'
    },
    {
      type: 'tool',
      name: 'youtube_extract_claims',
      params: { video_url: 'https://youtube.com/watch?v=HmQkMRZFd8s' },
      response: {
        label: '✓ 5 claims extracted',
        items: [
          { icon: '🔬', text: '"Isaac Zimmerman scaled plumbing from $5M to $30–40M with 75 hires" → data_or_metric · verifiable ✓' },
          { icon: '💡', text: '"The lump of labor fallacy assumes fixed work exists" → creator_opinion · verifiable ✗' },
          { icon: '🔮', text: '"AI agents will replace 80% of routine knowledge work" → market_forecast · verifiable ✓' }
        ]
      },
      progress: 72,
      label: 'Claims verified'
    },
    {
      type: 'tool',
      name: 'youtube_extract_recommendations',
      params: { video_url: 'https://youtube.com/watch?v=HmQkMRZFd8s' },
      response: {
        label: '✓ 5 recommendations extracted',
        items: [
          { icon: '✅', text: 'Build code-as-product leverage (stance: strongly positive · conviction: high)' },
          { icon: '✅', text: 'Hire globally via async remote talent (stance: positive · conviction: high)' },
          { icon: '⚠️', text: 'Reject revenue that requires linear headcount (stance: cautionary · conviction: high)' }
        ],
        citation: { time: '00:14:05', url: 'https://www.youtube.com/watch?v=HmQkMRZFd8s&t=845', quote: '"If adding a dollar of revenue requires adding a person, you don\'t have a business — you have a job with employees."' }
      },
      progress: 90,
      label: 'Recommendations extracted'
    },
    {
      type: 'assistant',
      text: '<strong>Research complete.</strong> Michael Bettan (@mbettan) advocates a 4-tier leverage hierarchy for solopreneurs: <strong>code → media → capital → labor</strong>. His core thesis is that labor is the lowest-leverage option, and solopreneurs should actively reject revenue that requires linear headcount. He recommends building code products with near-zero marginal cost, using AI agents as force multipliers, and hiring globally when human labor is needed. All 5 claims were verified against the video with timestamped citations.',
      progress: 100,
      label: 'Research complete · 5 tools used'
    }
  ];

  let isPlaying = false;
  let abortController = null;

  function sleep(ms) {
    return new Promise(resolve => {
      const id = setTimeout(resolve, ms);
      if (abortController) {
        abortController.signal.addEventListener('abort', () => { clearTimeout(id); resolve(); });
      }
    });
  }

  function scrollToBottom() {
    const lastChild = agentChat.lastElementChild;
    if (lastChild) lastChild.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function setProgress(pct, label) {
    if (progressBar) progressBar.style.width = pct + '%';
    if (progressLabel) progressLabel.textContent = label;
  }

  function buildToolHTML(step) {
    let paramsHTML = Object.entries(step.params).map(([k, v]) =>
      `<span class="param-key">${esc(k)}</span>: <span class="param-val">${esc(typeof v === 'string' ? '"' + v + '"' : String(v))}</span>`
    ).join('<br>');

    let responseHTML = `<div class="response-label">${step.response.label}</div><div class="response-items">`;
    step.response.items.forEach(item => {
      responseHTML += `<div class="response-item"><span class="item-icon">${item.icon}</span> ${esc(item.text)}</div>`;
    });

    if (step.response.citation) {
      const c = step.response.citation;
      responseHTML += `<a href="${esc(c.url)}" target="_blank" rel="noopener noreferrer" class="citation-link">▶ [Video @ ${esc(c.time)}]</a>`;
      responseHTML += `<div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.15rem;font-style:italic;">"${esc(c.quote)}"</div>`;
    }
    responseHTML += '</div>';

    return `<div class="agent-msg-tool">
      <div class="tool-call-header">
        <span class="tool-call-icon">⚡</span>
        <span class="tool-call-name">${esc(step.name)}</span>
        <span class="tool-call-badge">tubelens-oss MCP</span>
      </div>
      <div class="tool-call-params">${paramsHTML}</div>
      <div class="tool-call-response">${responseHTML}</div>
    </div>`;
  }

  async function runDemo() {
    if (isPlaying) return;
    isPlaying = true;
    abortController = new AbortController();

    agentChat.innerHTML = '';
    setProgress(0, 'Starting...');

    for (const step of steps) {
      if (abortController.signal.aborted) break;

      let el;
      if (step.type === 'user') {
        el = document.createElement('div');
        el.className = 'agent-msg-user';
        el.textContent = step.text;
      } else if (step.type === 'thinking') {
        el = document.createElement('div');
        el.className = 'agent-msg-thinking';
        el.innerHTML = `<div class="thinking-dots"><span></span><span></span><span></span></div> ${esc(step.text)}`;
      } else if (step.type === 'tool') {
        // Show thinking briefly first
        const thinkEl = document.createElement('div');
        thinkEl.className = 'agent-msg-thinking';
        thinkEl.innerHTML = `<div class="thinking-dots"><span></span><span></span><span></span></div> Calling ${esc(step.name)}...`;
        agentChat.appendChild(thinkEl);
        scrollToBottom();
        await sleep(600);
        if (abortController.signal.aborted) break;
        thinkEl.remove();

        el = document.createElement('div');
        el.innerHTML = buildToolHTML(step);
        el = el.firstElementChild;
      } else if (step.type === 'assistant') {
        el = document.createElement('div');
        el.className = 'agent-msg-assistant';
        el.innerHTML = step.text;
      }

      if (el) {
        agentChat.appendChild(el);
        scrollToBottom();
        setProgress(step.progress, step.label);
      }

      // Delay between steps
      const delay = step.type === 'tool' ? 1400 : step.type === 'thinking' ? 1000 : step.type === 'user' ? 800 : 500;
      await sleep(delay);
    }

    // Remove any lingering thinking bubbles
    agentChat.querySelectorAll('.agent-msg-thinking').forEach(el => {
      if (!el.textContent.includes('I\'ll start') && !el.textContent.includes('Found the') && !el.textContent.includes('The first')) return;
      // Keep the ones that are part of the narrative
    });

    isPlaying = false;
    abortController = null;
  }

  // Replay button
  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      if (abortController) abortController.abort();
      isPlaying = false;
      setTimeout(runDemo, 100);
    });
  }

  // Auto-start when section scrolls into view
  let hasAutoPlayed = false;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !hasAutoPlayed && !isPlaying) {
        hasAutoPlayed = true;
        runDemo();
      }
    });
  }, { threshold: 0.3 });

  const section = document.getElementById('simulator');
  if (section) observer.observe(section);
});
