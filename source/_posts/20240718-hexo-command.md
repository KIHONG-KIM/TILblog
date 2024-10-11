---
title: hexo 명령어 알아보기
date: 2024-07-18 09:21:11
categories:
- Hexo
tags: hexo
---

## Hexo 명령어

hexo에서 명령어를 작성할 때, 처음 해야할 일이 있는데 그것은 CLI로 명령어를 알아야 글을 작성하든, 글을 올리든 할 것이다.
첫 작성 포스트는 hexo 명령어에 대해 올려본다.
<!-- more -->
#### 1. 글 작성하기(post)


- 명령어 (command line)


```
hexo new [layout] <title>
```

- 설명

    layout: _config.yml 파일에 정의된 레이아웃 이름 (기본값: post)
    title: 생성할 포스트나 페이지의 제목

#### 2. generate

- hexo generate 명령어는 Markdown 형식으로 작성된 파일을 HTML로 변환하고, 정적 파일을 생성하는 데 사용됩니다. 
- 이 명령어를 통해 블로그의 소스 파일을 기반으로 글을 만들 수 있습니다.

```
hexo generate 
```

또는


```
hexo g
```

- 추가 옵션

    -d 또는 --deploy //  사이트를 생성한 후 자동으로 배포합니다.
    -w 또는 --watch // 파일 변화를 감지하여 자동으로 다시 생성합니다.



#### 3. 배포하기(deploy)

_config.yml에 등록한 깃허브 레포지토리로 배포합니다.


```
hexo deploy
```
- 추가 옵션
-g 또는 --generate: 배포 전에 generate 명령을 실행합니다.
-m 또는 --message: 커밋 메시지를 지정합니다(Git을 사용하는 경우)
--dry-run: 실제 배포 없이 테스트를 수행합니다.


요약하면

hexo generate: 사이트의 정적 파일을 생성합니다.
hexo new: 새로운 포스트나 페이지를 생성합니다.
hexo deploy: 생성된 사이트를 서버에 배포합니다.