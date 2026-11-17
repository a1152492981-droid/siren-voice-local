import { getSirenSettings } from "./settings.js";

// ==========================================
// [fork] 官方 GPT-SoVITS api_v2 兼容层
// 官方 api_v2 (api_v2.py) 只有 /tts、/set_gpt_weights、/set_sovits_weights，
// 没有 /infer_classic、/classic_model_list、/custom_refs、/upload。
// api_mode: "auto"(默认, 自动识别) | "classic"(原版行为) | "api_v2"
// ==========================================
const _gsvModeCache = {};

/**
 * 判断后端接口类型。返回 "api_v2" 或 "classic"。
 * auto 模式下读 /openapi.json：有 /infer_classic 走原版，只有 /tts 走 api_v2；读不到则按原版处理。
 */
export async function detectGsvApiMode(apiBase, mode, force = false) {
  if (mode === "api_v2" || mode === "classic") return mode;
  if (!force && _gsvModeCache[apiBase]) return _gsvModeCache[apiBase];
  let result = "classic";
  try {
    const res = await fetch(`${apiBase}/openapi.json`);
    if (res.ok) {
      const paths = (await res.json()).paths || {};
      if (!paths["/infer_classic"] && paths["/tts"] && paths["/set_gpt_weights"]) {
        result = "api_v2";
      }
    }
  } catch (e) {
    console.warn("[Siren GSV] 接口类型自动识别失败，按原版接口处理:", e.message);
  }
  _gsvModeCache[apiBase] = result;
  return result;
}

/**
 * 解析「api_v2 路径列表」：每行一个服务器上的绝对路径。
 * .ckpt → GPT 模型，.pth → SoVITS 模型，音频文件 → 参考音频；
 * 参考音频行可写成「路径|参考文本」，文本会在选中时自动填入。
 */
export function parseGsvApiV2List(raw) {
  const out = { gpt: [], sovits: [], refs: [], refTexts: {} };
  (raw || "").split(/\r?\n/).forEach((line) => {
    line = line.trim().replace(/^["']|["']$/g, "");
    if (!line || line.startsWith("#")) return;
    let [path, text] = line.split("|");
    path = path.trim().replace(/^["']|["']$/g, "");
    const lower = path.toLowerCase();
    if (lower.endsWith(".ckpt")) out.gpt.push(path);
    else if (lower.endsWith(".pth")) out.sovits.push(path);
    else if (/\.(wav|mp3|flac|ogg|m4a)$/.test(lower)) {
      out.refs.push(path);
      if (text && text.trim()) out.refTexts[path] = text.trim();
    }
  });
  return out;
}

// Siren 的中文选项 → 官方 api_v2 的语言代码
const GSV_V2_LANG = {
  中文: "all_zh",
  英语: "en",
  日语: "all_ja",
  粤语: "all_yue",
  韩语: "all_ko",
  中英混合: "zh",
  日英混合: "ja",
  粤英混合: "yue",
  韩英混合: "ko",
  多语种混合: "auto",
  "多语种混合(粤语)": "auto_yue",
};
const GSV_V2_SPLIT = {
  不切: "cut0",
  凑四句一切: "cut1",
  凑50字一切: "cut2",
  "按中文句号。切": "cut3",
  "按英文句号.切": "cut4",
  按标点符号切: "cut5",
};
const _gsvV2Loaded = {}; // apiBase -> { gpt, sovits }，避免每句都重新加载权重

async function _gsvV2Get(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const d = await res.json();
      msg = d.message || d.Exception || d.detail || JSON.stringify(d);
      if (typeof msg !== "string") msg = JSON.stringify(msg);
    } catch {}
    throw new Error(msg);
  }
}

async function generateGptSovitsAudioV2(apiBase, gsvSettings, p) {
  const loaded = _gsvV2Loaded[apiBase] || (_gsvV2Loaded[apiBase] = {});
  const always = !!gsvSettings.api_v2_always_set_weights;
  try {
    if (always || loaded.gpt !== p.gpt) {
      loaded.gpt = null;
      await _gsvV2Get(`${apiBase}/set_gpt_weights?weights_path=${encodeURIComponent(p.gpt)}`);
      loaded.gpt = p.gpt;
    }
    if (always || loaded.sovits !== p.sovits) {
      loaded.sovits = null;
      await _gsvV2Get(`${apiBase}/set_sovits_weights?weights_path=${encodeURIComponent(p.sovits)}`);
      loaded.sovits = p.sovits;
    }
  } catch (err) {
    const msg = err.message.startsWith("Failed to fetch")
      ? "无法连接到 GSV 后端（未启动，或 api_v2 没有开启跨域 CORS）。"
      : `切换模型失败: ${err.message}`;
    if (window.toastr) window.toastr.error(msg, "Siren Voice");
    throw new Error(msg);
  }

  const body = {
    text: p.text,
    text_lang: GSV_V2_LANG[p.textLang] || p.textLang || "all_zh",
    ref_audio_path: p.refPath,
    prompt_text: p.refText,
    prompt_lang: GSV_V2_LANG[p.promptLang] || p.promptLang || "all_zh",
    top_k: p.topk,
    top_p: p.topp,
    temperature: p.temp,
    text_split_method: GSV_V2_SPLIT[gsvSettings.text_split_method] || "cut5",
    batch_size: 1,
    batch_threshold: 0.75,
    split_bucket: true,
    speed_factor: p.speed,
    fragment_interval: gsvSettings.fragment_interval || 0.3,
    seed: p.seed,
    media_type: "wav",
    streaming_mode: false,
    parallel_infer: gsvSettings.parallel_infer ?? true,
    repetition_penalty: p.rep,
  };
  console.log("[Siren GSV] (api_v2) 发起请求 Payload:", body);

  let response;
  try {
    response = await fetch(`${apiBase}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (networkErr) {
    const errMsg = "无法连接到 GSV 后端（未启动，或 api_v2 没有开启跨域 CORS）。";
    if (window.toastr) window.toastr.error(errMsg, "网络错误");
    throw new Error(`网络连接失败: ${networkErr.message}`);
  }
  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const d = await response.json();
      errorMsg = d.message || d.Exception || d.detail || errorMsg;
      if (typeof errorMsg !== "string") errorMsg = JSON.stringify(errorMsg);
    } catch {}
    if (window.toastr) window.toastr.error(`推理失败: ${errorMsg}`, "Siren Voice");
    throw new Error(`推理失败: ${errorMsg}`);
  }
  const blob = await response.blob();
  if (blob.size === 0) throw new Error("接收到的音频流为空！");
  return blob;
}

/**
 * 辅助函数：获取当前角色卡里的 GSV 配置
 * (与 UI 层分离，保证逻辑层随时可以独立读取最新数据)
 */
function getGptSovitsCardData() {
  const context = SillyTavern.getContext();
  const charId = context.characterId;
  if (
    charId !== undefined &&
    charId !== null &&
    context.characters &&
    context.characters[charId]
  ) {
    return (
      context.characters[charId].data?.extensions?.siren_voice_gptsovits || {}
    );
  }
  return {};
}

/**
 * 核心推理函数：根据角色名和情绪名生成音频
 * * @param {string} text 要生成的 TTS 文本
 * @param {string} charName 角色名称 (用于匹配配置映射)
 * @param {string} emotionName 情绪名称 (可为空，为空则走角色默认设置)
 * @returns {Promise<Blob>} 成功则返回 Wav 音频文件流
 */
export async function generateGptSovitsAudio(text, charName, emotionName = "") {
  const settings = getSirenSettings();
  const gsvSettings = settings.tts?.gptsovits || {};
  const apiBase = (gsvSettings.api_base || "http://127.0.0.1:9880").replace(
    /\/$/,
    "",
  );

  // 获取当前角色卡里的映射表
  const cardData = getGptSovitsCardData();
  const chars = cardData.characters || [];
  const emos = cardData.emotions || [];

  // 1. 基底配置 (角色)
  const charConfig = chars.find((c) => c.charName === charName);
  if (!charConfig) {
    const errMsg = `未找到角色 [${charName}] 的 GSV 绑定配置，请前往扩展面板添加。`;
    if (window.toastr) window.toastr.warning(errMsg, "Siren Voice");
    throw new Error(errMsg);
  }

  // 2. 初始化全局兜底参数 (Fallback)
  const globalParams = {
    speed: parseFloat(gsvSettings.speed_factor) || 1.0,
    temp: parseFloat(gsvSettings.temperature) || 1.0,
    topp: parseFloat(gsvSettings.top_p) || 1.0,
    topk: parseInt(gsvSettings.top_k) || 15,
    rep: parseFloat(gsvSettings.repetition_penalty) || 1.35,
    seed: parseInt(gsvSettings.seed) || -1,
    textLang: "中文",
  };

  // 3. 决定最终发送状态
  let finalGptModel = charConfig.gptModel;
  let finalSovitsModel = charConfig.sovitsModel;
  let finalRefPath = charConfig.refPath;
  let finalRefText = charConfig.refText;
  let finalPromptLang = charConfig.promptLang || "中文";

  // 默认使用角色特有推理参数，如果为空，也用兜底填充
  let finalParams = {
    speed: charConfig.speedFactor ?? globalParams.speed,
    temp: charConfig.temperature ?? globalParams.temp,
    topp: charConfig.topP ?? globalParams.topp,
    topk: charConfig.topK ?? globalParams.topk,
    rep: charConfig.repetitionPenalty ?? globalParams.rep,
    seed: charConfig.seed ?? globalParams.seed,
    textLang: charConfig.textLang || "中文",
  };

  // 4. 情绪与降级核心逻辑
  if (emotionName) {
    // 🌟 新增：支持同名匹配 (愤怒, 愤怒) 或 数字后缀匹配 (愤怒-1, 愤怒-2)
    // 使用正则防止误判，比如避免 "愤怒" 意外匹配到 "愤怒的咆哮"
    const emotionRegex = new RegExp(`^${emotionName}(-\\d+)?$`);

    const matchingEmos = emos.filter(
      (e) => e.charName === charName && emotionRegex.test(e.emotion),
    );

    let emoConfig = null;
    if (matchingEmos.length > 0) {
      // 如果匹配到多个，则进行随机抽取
      const randomIndex = Math.floor(Math.random() * matchingEmos.length);
      emoConfig = matchingEmos[randomIndex];

      console.log(
        `[Siren GSV] 🎲 情绪池抽取: 找到 ${matchingEmos.length} 个 [${emotionName}] 的变体，随机命中: [${emoConfig.emotion}] (模式: ${emoConfig.mode})`,
      );
    }

    if (emoConfig) {
      // 匹配成功时，保留角色卡的推流参数 (finalParams 不变)
      if (emoConfig.mode === "audio") {
        // 情形3：音频控制。替换音频路径与文本
        if (emoConfig.refPath) finalRefPath = emoConfig.refPath;
        if (emoConfig.refText) finalRefText = emoConfig.refText;
      } else if (emoConfig.mode === "model") {
        // 替换 GPT 模型
        if (emoConfig.gptModel) finalGptModel = emoConfig.gptModel;

        // 情形4/5：判断用户是否输入了额外的独立参考音频
        if (emoConfig.refPath && emoConfig.refText) {
          finalRefPath = emoConfig.refPath;
          finalRefText = emoConfig.refText;
          finalPromptLang = emoConfig.promptLang || finalPromptLang;
        } else {
          console.log(
            `[Siren GSV] 模型控制未设置独立音频，降级使用角色默认参考音频`,
          );
        }
      }
    } else {
      // 情形2：情绪没匹配上 (降级措施)
      console.warn(
        `[Siren GSV] 降级触发：情绪 [${emotionName}] 未注册。启用全局默认推理参数与角色默认参考音频。`,
      );
      finalParams = {
        ...globalParams,
        textLang: charConfig.textLang || globalParams.textLang,
      };
    }
  }

  // 5. 校验阶段及路径智能补全
  if (!finalGptModel || !finalSovitsModel || !finalRefPath || !finalRefText) {
    const errMsg = `角色 [${charName}] 的模型或参考音频配置不完整，请检查。`;
    if (window.toastr) window.toastr.error(errMsg, "Siren Voice");
    throw new Error(errMsg);
  }

  // [fork] 官方 api_v2：参考音频填服务器上的绝对路径，不加 custom_refs/ 前缀
  if (
    (await detectGsvApiMode(apiBase, gsvSettings.api_mode || "auto")) ===
    "api_v2"
  ) {
    const strip = (m) => (m.includes("::") ? m.split("::").slice(1).join("::") : m);
    return generateGptSovitsAudioV2(apiBase, gsvSettings, {
      gpt: strip(finalGptModel),
      sovits: strip(finalSovitsModel),
      refPath: finalRefPath.trim(),
      refText: finalRefText,
      promptLang: finalPromptLang,
      text: text,
      textLang: finalParams.textLang,
      speed: finalParams.speed,
      temp: finalParams.temp,
      topp: finalParams.topp,
      topk: finalParams.topk,
      rep: finalParams.rep,
      seed: finalParams.seed,
    });
  }

  let processedRefPath = finalRefPath.trim();
  if (!processedRefPath.includes("/") && !processedRefPath.includes("\\")) {
    processedRefPath = `custom_refs/${processedRefPath}`;
  }

  let reqVersion = "v4";
  let realGptName = finalGptModel;
  if (finalGptModel.includes("::")) {
    [reqVersion, realGptName] = finalGptModel.split("::");
  }
  let realSovitsName = finalSovitsModel.includes("::")
    ? finalSovitsModel.split("::")[1]
    : finalSovitsModel;

  // 6. 构建严丝合缝的 Payload
  const payload = {
    app_key: gsvSettings.api_key || "",
    version: reqVersion,
    gpt_model_name: realGptName,
    sovits_model_name: realSovitsName,
    ref_audio_path: processedRefPath,
    prompt_text: finalRefText,
    prompt_text_lang: finalPromptLang,

    text: text,
    text_lang: finalParams.textLang,

    // 核心语音参数 (取自状态机)
    speed_facter: finalParams.speed,
    temperature: finalParams.temp,
    top_p: finalParams.topp,
    top_k: finalParams.topk,
    repetition_penalty: finalParams.rep,
    seed: finalParams.seed,

    // 全局级控制参数
    text_split_method: gsvSettings.text_split_method || "按标点符号切",
    fragment_interval: gsvSettings.fragment_interval || 0.3,
    parallel_infer: gsvSettings.parallel_infer ?? true,

    batch_size: 1,
    batch_threshold: 0.75,
    split_bucket: true,
    sample_steps: 16,
    if_sr: false,
    media_type: "wav",
  };

  console.log("[Siren GSV] 发起请求 Payload:", payload);

  // 6. 发起 HTTP 请求
  let response;
  try {
    response = await fetch(`${apiBase}/infer_classic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (networkErr) {
    // 这里捕获的是网络层面的报错，比如后端根本没启动、跨域被浏览器拦截、或者是地址写错了
    const errMsg = "无法连接到 GSV 后端，请确认服务已启动且地址正确。";
    console.error("[Siren GSV] 网络请求失败:", networkErr);
    if (window.toastr) window.toastr.error(errMsg, "网络错误");
    // 抛出错误，中断后续流程，外层的 catch 会捕获并重置 UI 按钮状态
    throw new Error(`网络连接失败: ${networkErr.message}`);
  }

  if (!response.ok) {
    // 如果后端报错，尝试提取详细的报错原因
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errData = await response.json();
      // 优先读取你后端返回的 msg 字段
      if (errData.msg) errorMsg = errData.msg;
      else if (errData.detail) errorMsg = JSON.stringify(errData.detail);
    } catch {
      const errText = await response.text();
      if (errText) errorMsg = errText;
    }

    // 针对 API Key 错误单独弹窗
    if (response.status === 401) {
      const msg = "API Key 错误或未授权，请检查 GSV 接口设置！";
      if (window.toastr) window.toastr.error(msg, "Siren Voice");
      throw new Error(msg);
    } else {
      if (window.toastr)
        window.toastr.error(`推理失败: ${errorMsg}`, "Siren Voice");
      throw new Error(`推理失败: ${errorMsg}`);
    }
  }

  // 7. 由于你修改了后端直接返回 Response(content=audio_byte, media_type="audio/wav")
  // 我们直接将其解析为 Blob
  const blob = await response.blob();
  if (blob.size === 0) {
    throw new Error("接收到的音频流为空！");
  }

  return blob;
}
