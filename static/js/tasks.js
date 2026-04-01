/**
 * Task Management Logic (Kanban)
 */

document.addEventListener('DOMContentLoaded', function () {
    loadTasks();
});

// --- Load & Render ---
async function loadTasks() {
    try {
        const res = await fetch('/api/tasks');
        const result = await res.json();

        if (!result.success) return;

        const tasks = result.data;

        // Reset Columns (use class selectors matching the HTML template)
        const colPlanned = document.querySelector('.column-todo .kanban-body');
        const colProcess = document.querySelector('.column-progress .kanban-body');
        const colComplete = document.querySelector('.column-done .kanban-body');
        if (!colPlanned || !colProcess || !colComplete) return;

        colPlanned.innerHTML = '';
        colProcess.innerHTML = '';
        colComplete.innerHTML = '';

        let countPlanned = 0, countProcess = 0, countComplete = 0;

        tasks.forEach(task => {
            const card = createTaskCard(task);

            if (task.status === 'planned') {
                colPlanned.appendChild(card);
                countPlanned++;
            } else if (task.status === 'in_progress') {
                colProcess.appendChild(card);
                countProcess++;
            } else if (task.status === 'completed') {
                colComplete.appendChild(card);
                countComplete++;
            } else {
                // Default to planned
                colPlanned.appendChild(card);
                countPlanned++;
            }
        });

        // Update Counts (use class selectors matching the HTML template)
        const cntPlanned = document.querySelector('.column-todo .kanban-count');
        const cntProcess = document.querySelector('.column-progress .kanban-count');
        const cntComplete = document.querySelector('.column-done .kanban-count');
        if (cntPlanned) cntPlanned.innerText = countPlanned;
        if (cntProcess) cntProcess.innerText = countProcess;
        if (cntComplete) cntComplete.innerText = countComplete;

    } catch (e) {
        console.error("Task load failed", e);
    }
}

function createTaskCard(task) {
    const div = document.createElement('div');
    div.className = `task-card status-${task.status === 'in_progress' ? 'process' : (task.status === 'completed' ? 'complete' : 'planned')}`;

    // Status badges or Move buttons
    // Logic: 
    // Planned -> In Progress
    // In Progress -> Completed, Back to Planned
    // Completed -> Back to Process

    let actions = '';
    if (task.status === 'planned') {
        actions = `<button class="action-btn" onclick="updateTaskStatus(${task.id}, 'in_progress')" title="Start"><i class="bi bi-play-fill"></i></button>`;
    } else if (task.status === 'in_progress') {
        actions = `
            <button class="action-btn" onclick="updateTaskStatus(${task.id}, 'planned')" title="Back"><i class="bi bi-arrow-left"></i></button>
            <button class="action-btn" onclick="updateTaskStatus(${task.id}, 'completed')" title="Done"><i class="bi bi-check"></i></button>
        `;
    } else if (task.status === 'completed') {
        actions = `<button class="action-btn" onclick="updateTaskStatus(${task.id}, 'in_progress')" title="Reopen"><i class="bi bi-arrow-counterclockwise"></i></button>`;
    }

    div.innerHTML = `
        <div class="task-title">${task.title}</div>
        <div class="task-desc">${task.description || ''}</div>
        <div class="task-meta">
            <span>${new Date(task.created_at).toLocaleDateString()}</span>
            <div class="task-actions">${actions}</div>
        </div>
    `;
    return div;
}

// --- Actions ---
async function updateTaskStatus(id, newStatus) {
    try {
        const res = await fetch(`/api/tasks/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });
        const result = await res.json();
        if (result.success) {
            loadTasks(); // Reload
        } else {
            alert("상태 업데이트 실패: " + result.error);
        }
    } catch (e) {
        console.error(e);
    }
}

// --- Add Task Modal ---
let addTaskModal;
function openAddTaskModal(status = 'planned') {
    document.getElementById('taskStatus').value = status;
    document.getElementById('taskTitle').value = '';
    document.getElementById('taskDesc').value = '';

    const el = document.getElementById('addTaskModal');
    if (!addTaskModal) {
        addTaskModal = new bootstrap.Modal(el);
    }
    addTaskModal.show();
}

async function submitNewTask() {
    const title = document.getElementById('taskTitle').value;
    const desc = document.getElementById('taskDesc').value;
    const status = document.getElementById('taskStatus').value;

    if (!title) {
        alert("제목을 입력해주세요.");
        return;
    }

    try {
        const res = await fetch('/api/tasks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: title, description: desc, status: status })
        });
        const result = await res.json();

        if (result.success) {
            if (addTaskModal) addTaskModal.hide();
            loadTasks();
        } else {
            alert("작업 추가 실패");
        }
    } catch (e) {
        console.error(e);
    }
}
