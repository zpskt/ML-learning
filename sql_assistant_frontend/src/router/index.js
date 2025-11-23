import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import QueryView from '../views/QueryView.vue'
import HistoryView from '../views/HistoryView.vue'
import KnowledgeBaseView from '../views/KnowledgeBaseView.vue'
import ConfigView from '../views/ConfigView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/query',
    name: 'query',
    component: QueryView
  },
  {
    path: '/history',
    name: 'history',
    component: HistoryView
  },
  {
    path: '/knowledge',
    name: 'knowledge',
    component: KnowledgeBaseView
  },
  {
    path: '/config',
    name: 'config',
    component: ConfigView
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router