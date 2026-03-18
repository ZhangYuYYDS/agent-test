import { createAgent } from "langchain";
import { ChatDeepSeek } from "@langchain/deepseek";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import express from 'express'
import cors from 'cors'

const key = "sk-76fc3eb589a645f2bb97b9003931f6e3";
const bocha = "sk-4faff1f9f024488e93442dd46b7c1517";

const app = express()
app.use(cors())
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const bochaAPI = async (query: string) => {
    const response = await fetch('https://api.bochaai.com/v1/web-search', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${bocha}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query, summary: true, count: 10 })
    })
    const data = await response.json()
    const list = data.data.webPages.value
    return list.map((item: any) => item.summary).join('\n')
}

// 把搜索能力封装成 Tool，Agent 自主决定是否调用
const searchTool = tool(
    async (input) => {
        const result = await bochaAPI(input.query)
        console.log('[Tool 被调用] 搜索:', input.query)
        return result
    },
    {
        name: "web_search",
        description: "搜索互联网获取最新信息。当用户的问题涉及实时数据、最新新闻、具体事实或你不确定的知识时，使用此工具搜索。",
        schema: z.object({
            query: z.string().describe("搜索关键词"),
        }),
    }
)

const model = new ChatDeepSeek({
    model: "deepseek-chat",
    apiKey: key,
    temperature: 1.3,
    streaming: true,
});

// Agent 只创建一次，tools 已绑定，每次请求复用
const agent = createAgent({
    model,
    tools: [searchTool],
    systemPrompt: "今天有什么科技新闻吗？",
});

app.post('/', async (req, res) => {
    const msg = req.body.msg as string

    try {
        const result = await agent.invoke({
            messages: [{ role: "user", content: msg }],
        });
        const lastMessage = result.messages.at(-1)
        res.json({ reply: lastMessage?.content ?? '' })
    } catch (error) {
        console.log(error);
        res.status(500).json({ error: '请求失败' })
    }
})

app.listen(3000, () => {
    console.log('Server is running on port 3000')
})
