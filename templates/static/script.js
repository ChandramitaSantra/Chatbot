document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const assetIdDisplay = document.getElementById('assetIdDisplay');
    const assetIdSpan = document.getElementById('assetId');
    const chatSection = document.getElementById('chatSection');
    const chatContainer = document.getElementById('chatContainer');
    const chatMessages = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');

    async function startChat(asset_id) {
        try {
            const response = await fetch('/api/chat/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ asset_id: asset_id })
            });

            const data = await response.json();
            if (response.ok) {
                return data.chat_id; // Return the chat_id for further use
            } else {
                alert(data.error || 'Failed to start chat.');
                return null;
            }
        } catch (error) {
            console.error('Error starting chat:', error);
            alert('Error starting chat.');
            return null;
        }
    }

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/api/documents/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                assetIdSpan.textContent = data.asset_id;
                assetIdDisplay.classList.remove('hidden');
                chatSection.classList.remove('hidden');

                const chat_id = await startChat(data.asset_id);

                if (chat_id) {
                    assetIdSpan.dataset.chatId = chat_id; // Store chat_id for further use
                }
            } else {
                alert(data.error || 'Failed to upload document.');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file.');
        }
    });

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const message = chatInput.value.trim();

        if (!message) return;

        const chat_id = assetIdSpan.dataset.chatId; // Retrieve the chat_id

        if (!chat_id) {
            alert('Chat session not initialized.');
            return;
        }

        try {
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    chat_id: chat_id,
                    message: message
                })
            });

            const data = await response.json();

            if (response.ok) {
                const userMessage = document.createElement('div');
                userMessage.textContent = `You: ${message}`;
                userMessage.className = 'message user-message';
                chatMessages.appendChild(userMessage);

                const botMessage = document.createElement('div');
                botMessage.textContent = `Bot: ${data.response}`;
                botMessage.className = 'message agent-message';
                chatMessages.appendChild(botMessage);

                chatInput.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to the bottom
            } else {
                alert(data.error || 'Failed to send message.');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            alert('Error sending message.');
        }
    });
});
