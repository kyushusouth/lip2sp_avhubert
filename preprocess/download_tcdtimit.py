import requests

def main():
    email = 'minami.taisuke.326@s.kyushu-u.ac.jp'
    password = 'tmnm13009'

    session = requests.Session()
    login_data = {
        'email': email,
        'password': password
    }
    login_url = 'https://sigmedia.tcd.ie/users/sign_in'
    response = session.post(login_url, data=login_data, verify=False)
    if response.status_code == 200:
        print('login successed')
    else:
        print('login failed')
        exit()

    download_url = 'https://sigmedia.tcd.ie/tcd_timit_db/sample_30_straight_video.mp4'
    download_response = session.get(download_url)
    if download_response.status_code == 200:
        with open('test.mp4', 'wb') as f:
            f.write(download_response.content)
        print('download successed')
    else:
        print('download failed')
    # download_response = session.get(download_url, stream=True)
    # if download_response.status_code == 200:
    #     with open('test.mp4', 'wb') as f:
    #         for chunk in download_response.iter_content(chunk_size=8192):
    #             f.write(chunk)    
    #     print('download successed')
    # else:
    #     print('download failed')



if __name__ == '__main__':
    main()