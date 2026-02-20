document.addEventListener('DOMContentLoaded', () => {
    // --- Global State ---
    let currentChatId = null;
    let modelStatus = "IDLE";

    // --- Elements ---
    const chatList = document.getElementById('chat-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatContainer = document.getElementById('chat-container');
    const textInput = document.getElementById('text-input');
    const generateBtn = document.getElementById('generate-btn');
    const voiceSelect = document.getElementById('voice-select');
    const modelSelect = document.getElementById('model-select');
    const modelLoader = document.getElementById('model-loader');
    const modelStatusText = document.getElementById('model-status');

    // --- Initialization ---
    loadChats(); // Always load chats for sidebar

    // Check if we are on the main chat page
    if (chatContainer) {
        checkModelStatus();
        loadVoicesForDropdown();
    }

    // Global listeners (sidebar)
    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewChat);
    }

    // Chat page listeners
    if (chatContainer) {
        generateBtn.addEventListener('click', sendMessage);

        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        modelSelect.addEventListener('change', (e) => {
            const selected = e.target.value;
            if (selected === 'custom') {
                // Show modal
                const modal = document.getElementById('custom-model-modal');
                if (modal) modal.classList.remove('hidden');
            } else if (selected) {
                loadModel(selected);
            }
        });

        // Custom model modal logic
        const customModelModal = document.getElementById('custom-model-modal');
        if (customModelModal) {
            document.getElementById('cancel-custom-model').addEventListener('click', () => {
                customModelModal.classList.add('hidden');
                modelSelect.value = "";
            });
            document.getElementById('load-custom-model').addEventListener('click', () => {
                const input = document.getElementById('custom-model-input');
                if (input.value) {
                    loadModel(input.value);
                    customModelModal.classList.add('hidden');
                }
            });
        }
    }

    // --- Functions ---

    async function loadChats() {
        try {
            const response = await fetch('/api/chats');
            const chats = await response.json();

            chatList.innerHTML = '';
            chats.forEach(chat => {
                const item = document.createElement('div');
                item.className = `p-3 rounded-lg cursor-pointer hover:bg-gray-800 transition-colors flex items-center gap-3 ${currentChatId === chat.id ? 'bg-gray-800' : ''}`;
                item.onclick = () => loadChat(chat.id);
                item.innerHTML = `
                    <i data-lucide="message-square" class="w-4 h-4 text-gray-400"></i>
                    <span class="truncate text-sm font-medium text-gray-200">${chat.title}</span>
                `;
                chatList.appendChild(item);
            });
            lucide.createIcons();
        } catch (error) {
            console.error('Failed to load chats:', error);
        }
    }

    async function createNewChat() {
        try {
            const response = await fetch('/api/chats', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: 'New Chat' })
            });
            const chat = await response.json();
            currentChatId = chat.id;
            loadChats();
            loadChat(chat.id);
        } catch (error) {
            console.error('Failed to create chat:', error);
        }
    }

    async function loadChat(chatId) {
        if (!chatContainer) {
            // If on voices page, redirect to home with chat id?
            // Or just redirect to home and let it load?
            // For simplicity, just redirect to /
            window.location.href = '/';
            // But how to load specific chat after redirect?
            // Maybe store in localStorage or query param.
            // For now, simple redirect. User can click again.
            return;
        }

        currentChatId = chatId;

        // Update active state in sidebar
        Array.from(chatList.children).forEach(child => {
            child.classList.remove('bg-gray-800');
        });

        chatContainer.innerHTML = ''; // Clear

        try {
            const response = await fetch(`/api/chats/${chatId}/messages`);
            const messages = await response.json();

            if (messages.length === 0) {
                chatContainer.innerHTML = `
                    <div class="flex flex-col items-center justify-center h-full text-gray-400">
                        <i data-lucide="message-square" class="w-12 h-12 mb-2 opacity-50"></i>
                        <p>No messages yet. Start chatting!</p>
                    </div>
                `;
            } else {
                messages.forEach(msg => appendMessage(msg));
            }
            scrollToBottom();
            lucide.createIcons();
        } catch (error) {
            console.error('Failed to load messages:', error);
        }
    }

    async function sendMessage() {
        const text = textInput.value.trim();
        if (!text || !currentChatId) return;

        // Optimistically add user message
        const userMsg = {
            role: 'user',
            content: text,
            timestamp: new Date().toISOString()
        };
        appendMessage(userMsg);
        textInput.value = '';
        scrollToBottom();

        // Show loading state for assistant
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.className = 'flex items-start gap-4 mb-6 animate-pulse';
        loadingDiv.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                <i data-lucide="bot" class="w-5 h-5 text-blue-600"></i>
            </div>
            <div class="flex-1 space-y-2">
                <div class="h-4 bg-gray-200 rounded w-1/4"></div>
                <div class="h-4 bg-gray-200 rounded w-3/4"></div>
            </div>
        `;
        chatContainer.appendChild(loadingDiv);
        scrollToBottom();
        lucide.createIcons();

        try {
            const voiceId = voiceSelect.value;
            const temperature = parseFloat(document.getElementById('temperature').value);
            const top_p = parseFloat(document.getElementById('top_p').value);

            const response = await fetch(`/api/chats/${currentChatId}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chat_id: currentChatId,
                    content: text,
                    voice_id: voiceId || null,
                    temperature: temperature,
                    top_p: top_p
                })
            });

            // Remove loading
            document.getElementById(loadingId).remove();

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Generation failed');
            }

            const assistantMsg = await response.json();
            appendMessage(assistantMsg);

            // Auto-play audio
            if (assistantMsg.audio_url) {
                const audio = new Audio(assistantMsg.audio_url);
                audio.play();
            }

        } catch (error) {
             document.getElementById(loadingId)?.remove();
             alert(`Error: ${error.message}`);
        }
        scrollToBottom();
    }

    function appendMessage(msg) {
        const isUser = msg.role === 'user';

        // Remove "empty" placeholder if present
        if (chatContainer.querySelector('p')?.textContent.includes('Start chatting')) {
            chatContainer.innerHTML = '';
        }

        const div = document.createElement('div');
        div.className = `flex items-start gap-4 mb-6 ${isUser ? 'flex-row-reverse' : ''}`;

        const avatar = isUser
            ? `<div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0"><i data-lucide="user" class="w-5 h-5 text-gray-600"></i></div>`
            : `<div class="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0"><i data-lucide="bot" class="w-5 h-5 text-blue-600"></i></div>`;

        const contentClass = isUser
            ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-md max-w-[80%]'
            : 'bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm max-w-[80%]';

        let audioPlayer = '';
        if (msg.audio_url) {
            audioPlayer = `
                <div class="mt-3 flex items-center gap-2 bg-gray-50 rounded-lg p-2 border border-gray-200">
                    <button class="w-8 h-8 rounded-full bg-blue-600 hover:bg-blue-700 flex items-center justify-center text-white transition-colors" onclick="new Audio('${msg.audio_url}').play()">
                        <i data-lucide="play" class="w-4 h-4 fill-current"></i>
                    </button>
                    <div class="text-xs text-gray-500 font-mono">Audio Generated</div>
                    <a href="${msg.audio_url}" download class="ml-auto text-gray-400 hover:text-blue-600" title="Download Audio">
                        <i data-lucide="download" class="w-4 h-4"></i>
                    </a>
                </div>
            `;
        }

        div.innerHTML = `
            ${avatar}
            <div class="${contentClass}">
                <p class="whitespace-pre-wrap leading-relaxed">${msg.content}</p>
                ${audioPlayer}
            </div>
        `;

        chatContainer.appendChild(div);
        lucide.createIcons();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function loadVoicesForDropdown() {
        try {
            const response = await fetch('/api/voices');
            const voices = await response.json();

            // Clear but keep default
            voiceSelect.innerHTML = '<option value="">Default Voice</option>';

            voices.forEach(voice => {
                const opt = document.createElement('option');
                opt.value = voice.id;
                opt.textContent = voice.name;
                voiceSelect.appendChild(opt);
            });
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }

    async function checkModelStatus() {
        // Poll status every 2 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/api/model/status');
                const status = await response.json();

                modelStatusText.textContent = `Status: ${status.status}`;
                if (status.model_name) {
                     modelStatusText.textContent += ` (${status.model_name})`;
                }

                if (status.status === 'LOADING') {
                    modelLoader.classList.remove('hidden');
                    modelSelect.disabled = true;
                } else {
                    modelLoader.classList.add('hidden');
                    modelSelect.disabled = false;
                }
            } catch (error) {
                console.error('Status check failed', error);
            }
        }, 2000);
    }

    async function loadModel(modelName) {
        try {
            modelSelect.disabled = true;
            modelLoader.classList.remove('hidden');
            modelStatusText.textContent = "Requesting load...";

            const response = await fetch('/api/model/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            });

            if (!response.ok) {
                throw new Error('Failed to start loading');
            }

            // Polling will update UI
        } catch (error) {
            alert(error.message);
            modelSelect.disabled = false;
            modelLoader.classList.add('hidden');
        }
    }
});
