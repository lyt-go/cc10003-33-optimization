<template>
  <section class="page" data-module="forecast">
    <header class="page-head">
      <div>
        <h2>预测模型管理</h2>
        <p class="page-desc">管理时序预测模型方案，配置预测目标、算法选择和特征工程参数。</p>
      </div>
      <div class="page-actions">
        <button class="btn primary" type="button" @click="openCreate">登记预测方案</button>
        <button class="btn" type="button" @click="exportRows">导出预测模型清单</button>
      </div>
    </header>

    <div class="stat-row">
      <article v-for="item in stats" :key="item.label" class="stat-card">
        <span class="stat-label">{{ item.label }}</span>
        <strong class="stat-value">{{ item.value }}</strong>
      </article>
    </div>

    <form class="filter-bar" @submit.prevent="reload">
      <label v-for="field in filterFields" :key="field" class="filter-item">
        <span>{{ field }}</span>
        <input v-model="filters[field]" :placeholder="`按${field}检索`" />
      </label>
      <button class="btn" type="submit">查询</button>
      <button class="btn ghost" type="button" @click="resetFilters">重置条件</button>
    </form>

    <table class="data-table">
      <thead>
        <tr>
          <th v-for="column in columns" :key="column">{{ column }}</th>
          <th>可执行动作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="String(row.id)">
          <td v-for="column in columns" :key="column">{{ row[column] ?? '—' }}</td>
          <td class="row-actions">
            <button
              v-for="action in actions"
              :key="action"
              class="link"
              type="button"
              @click="runAction(action, row)"
            >
              {{ action }}
            </button>
            <button class="link" type="button" @click="executeRow(row)">提交执行</button>
          </td>
        </tr>
        <tr v-if="!rows.length">
          <td :colspan="columns.length + 1" class="empty-state">暂无预测模型数据，可先登记预测方案</td>
        </tr>
      </tbody>
    </table>

    <section class="exec-panel">
      <h3>训练产物（artifact）</h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>模型版本</th>
            <th>artifact_sha256</th>
            <th>训练区间</th>
            <th>预测区间</th>
            <th>置信水平</th>
            <th>最新参数复评</th>
            <th>重算范围</th>
            <th>切片</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in artifacts" :key="item.artifact_sha256">
            <td>{{ item.model_version }}</td>
            <td :title="item.artifact_sha256">{{ shortSha(item.artifact_sha256) }}</td>
            <td>{{ spanText(item.train_span) }}</td>
            <td>{{ spanText(item.forecast_span) }}</td>
            <td>{{ item.confidence_level ?? '—' }}</td>
            <td>{{ item.latest_eval_version ?? '—' }}</td>
            <td>{{ scopeText(item.recompute_scope) }}</td>
            <td>{{ item.slices?.length ?? 0 }} 个</td>
          </tr>
          <tr v-if="!artifacts.length">
            <td colspan="8" class="empty-state">暂无训练产物，提交执行后在此查看</td>
          </tr>
        </tbody>
      </table>

      <h3>执行台账（run_ledger）</h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>run_id</th>
            <th>执行方式</th>
            <th>状态</th>
            <th>checkpoint_seq</th>
            <th>artifact</th>
            <th>模型版本</th>
            <th>复评版本</th>
            <th>更新时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="run in ledgerRows" :key="run.run_id">
            <td>{{ run.run_id }}</td>
            <td>{{ modeLabel(run.exec_mode) }}</td>
            <td>{{ statusLabel(run.status) }}</td>
            <td>{{ run.checkpoint_seq }} / {{ run.total_slices }}</td>
            <td :title="run.artifact_sha256">{{ shortSha(run.artifact_sha256) }}</td>
            <td>{{ run.model_version ?? '—' }}</td>
            <td>{{ run.eval_version ?? '—' }}</td>
            <td>{{ run.updated_at }}</td>
          </tr>
          <tr v-if="!ledgerRows.length">
            <td colspan="8" class="empty-state">暂无执行记录</td>
          </tr>
        </tbody>
      </table>
    </section>

    <footer class="page-foot">
      <span>共 {{ total }} 条预测模型记录</span>
      <span v-if="noticeMessage" class="notice-text">{{ noticeMessage }}</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { request } from '@/api/client'

type Row = Record<string, string | number | null>
type ExecRow = Record<string, any>

const ENDPOINT = '/api/forecast'
const columns = ["方案编号", "预测目标", "算法名称", "特征字段", "训练周期", "预测窗口天", "置信水平", "模型版本"]
const actions = ["提交模型评估", "部署模型", "下线模型"]
const statuses = ["草稿", "模型评估中", "已部署", "已下线"]
const stats = [{"label": "预测方案总数", "value": 0}, {"label": "已部署方案", "value": 0}, {"label": "模型评估中", "value": 0}]

const rows = ref<Row[]>([])
const total = ref(0)
const errorMessage = ref('')
const noticeMessage = ref('')
const filters = ref<Record<string, string>>({})
const filterFields = columns.slice(0, 3)

const artifacts = ref<ExecRow[]>([])
const ledgerRows = ref<ExecRow[]>([])
const MODE_LABELS: Record<string, string> = {
  cache_reuse: '缓存复用',
  train_new: '重新训练',
  parameter_replay: '参数复评',
  resumed_slice: '断点续跑',
}
const STATUS_LABELS: Record<string, string> = {
  running: '运行中',
  succeeded: '成功',
  failed: '失败',
}

function modeLabel(mode?: unknown) {
  return MODE_LABELS[String(mode ?? '')] ?? mode ?? '—'
}

function statusLabel(status?: unknown) {
  return STATUS_LABELS[String(status ?? '')] ?? status ?? '—'
}

function shortSha(sha?: unknown) {
  const text = String(sha ?? '')
  return text ? `${text.slice(0, 12)}…` : '—'
}

function spanText(span?: { start?: string; end?: string } | null) {
  return span?.start ? `${span.start} ~ ${span.end}` : '—'
}

function scopeText(scope?: { from?: string; to?: string } | null) {
  return scope?.from ? `${scope.from} ~ ${scope.to}` : '未重算'
}

async function executeRow(row: Row) {
  errorMessage.value = ''
  noticeMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/${row.id}/execute`, {
      method: 'POST',
      body: JSON.stringify({ values: {} }),
    })
    const payload = await response.json()
    if (!payload.ok) {
      throw new Error(payload.message || '执行未生效')
    }
    noticeMessage.value = payload.message
    await Promise.all([reload(), reloadExec()])
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '提交执行失败'
  }
}

async function reloadExec() {
  try {
    const [artifactRes, ledgerRes] = await Promise.all([
      request(`${ENDPOINT}/artifacts`),
      request(`${ENDPOINT}/ledger`),
    ])
    if (artifactRes.ok) {
      const payload = await artifactRes.json()
      artifacts.value = payload.items ?? []
    }
    if (ledgerRes.ok) {
      const payload = await ledgerRes.json()
      ledgerRows.value = payload.items ?? []
    }
  } catch {
    // 面板数据加载失败不阻塞主列表
  }
}

function resetFilters() {
  filters.value = {}
  void reload()
}

function exportRows() {
  window.open(`${ENDPOINT}/export`, '_blank')
}

function openCreate() {
  errorMessage.value = '预测方案登记入口尚未接入审批流'
}

async function runAction(action: string, row: Row) {
  errorMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/${row.id}/actions`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    })
    if (!response.ok) {
      throw new Error('预测模型动作未生效，请稍后重试')
    }
    await reload()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '预测模型操作失败'
  }
}

async function reload() {
  errorMessage.value = ''
  const query = new URLSearchParams(filters.value as Record<string, string>).toString()
  try {
    const response = await request(`${ENDPOINT}?${query}`)
    if (!response.ok) {
      throw new Error('预测方案列表读取失败')
    }
    const payload = await response.json()
    rows.value = payload.items ?? []
    total.value = payload.total ?? rows.value.length
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '预测模型列表读取失败'
  }
}

onMounted(() => {
  void reload()
  void reloadExec()
})
</script>
