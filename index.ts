import 'dotenv/config'
import { createAgent } from "langchain";
import { ChatDeepSeek } from "@langchain/deepseek";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import express from 'express'
import cors from 'cors'

const key = process.env.DEEPSEEK_API_KEY;
const bocha = process.env.BOCHA_API_KEY;
const visionKey = process.env.VISION_API_KEY;

const app = express()
app.use(cors())
app.use(express.json({ limit: '50mb' }))
app.use(express.urlencoded({ extended: true, limit: '50mb' }))

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

const visionAPI = async (imageUrl: string, question: string) => {
    const response = await fetch('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${visionKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: "qwen-vl-plus",
            messages: [
                {
                    role: "user",
                    content: [
                        { type: "image_url", image_url: { url: imageUrl } },
                        { type: "text", text: question }
                    ]
                }
            ]
        })
    })
    const data = await response.json()
    if (data.error) {
        console.log('[视觉API 错误]', data.error)
        return `图片识别失败: ${data.error.message || JSON.stringify(data.error)}`
    }
    return data.choices?.[0]?.message?.content ?? '图片识别未返回结果'
}

const searchTool = tool(
    async (input) => {
        const result = await bochaAPI(input.query)
        console.log('[Tool 搜索被调用] 搜索:', input.query)
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

const imageRecognitionTool = tool(
    async (input) => {
        console.log('[Tool 识图被调用] 识图:', input.imageUrl)
        const result = await visionAPI(input.imageUrl, input.question)
        return result
    },
    {
        name: "image_recognition",
        description: "识别和分析图片内容。当用户提供了图片URL并要求查看、识别、描述或分析图片时，使用此工具。",
        schema: z.object({
            imageUrl: z.string().describe("图片的URL地址"),
            question: z.string().describe("用户关于这张图片的问题，例如：图片里有什么、这是什么东西"),
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
    tools: [searchTool, imageRecognitionTool],
    systemPrompt: "你是一个智能助手，可以搜索互联网信息，也可以识别分析图片，请用中文回答用户的问题。",
});

app.post('/', async (req, res) => {
    const { msg, image } = req.body as { msg: string; image?: string }

    try {
        let content = msg

        if (image) {
            console.log('[识图] 检测到图片，调用视觉 API...')
            const imageDescription = await visionAPI(image, msg || '请描述这张图片的内容')
            console.log('[识图] 结果:', imageDescription)
            content = `用户的问题: ${msg || '请描述这张图片'}\n\n图片识别结果: ${imageDescription}`
        }

        const result = await agent.invoke({
            messages: [{ role: "user", content }],
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
