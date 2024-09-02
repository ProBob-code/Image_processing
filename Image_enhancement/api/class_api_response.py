from flask import make_response

class ApiResponse:
    
    def __init__(self):
        pass

    def responseSuccess(self, message, data={}, headers={}):
        return self.sendResponse(200, message, data, headers)

    def responseCreated(self, message, data={}, headers={}):
        return self.sendResponse(201, message, data, headers)
    
    def responseBadRequest(self, message, data={}, headers={}):
        return self.sendResponse(400, message, data, headers)
    
    def responseNotFound(self, message, data={}, headers={}):
        return self.sendResponse(404, message, data, headers)
    
    def responseServerError(self, message, data={}, headers={}):
        return self.sendResponse(500, message, data, headers)

    def sendResponse(self, statusCode, message, data={}, headers={}):
        return make_response({
            "status_code"   : statusCode,
            "message"       : message,
            "data"          : data
        }, statusCode, headers)