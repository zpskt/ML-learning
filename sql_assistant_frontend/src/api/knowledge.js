import axios from 'axios'
import API_CONFIG from '../config/api'

// 获取知识库条目
export const getKnowledge = (id = null) => {
  if (id) {
    return axios.get(`${API_CONFIG.KNOWLEDGE}?knowledge_id=${id}`)
  }
  return axios.get(API_CONFIG.KNOWLEDGE)
}

// 添加知识库条目
export const addKnowledge = (knowledgeData) => {
  return axios.post(API_CONFIG.KNOWLEDGE, knowledgeData)
}

// 更新知识库条目
export const updateKnowledge = (knowledgeData) => {
  return axios.put(API_CONFIG.KNOWLEDGE, knowledgeData)
}

// 删除知识库条目
export const deleteKnowledge = (id) => {
  return axios.delete(API_CONFIG.KNOWLEDGE, { data: { id } })
}

export default {
  getKnowledge,
  addKnowledge,
  updateKnowledge,
  deleteKnowledge
}